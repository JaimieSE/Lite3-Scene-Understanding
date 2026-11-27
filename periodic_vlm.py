#!/usr/bin/env python3
"""Direct RTSP -> OpenCV -> periodic Claude observations. No robot control."""
import argparse
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import threading
import time

import anthropic
import urllib.request
import urllib.error
from types import SimpleNamespace
import cv2

ROOT = Path(__file__).resolve().parent
URL = 'rtsp://192.168.1.120:8554/test'
REQUIRED = {'scene', 'hazard_signs', 'people', 'stairs', 'curtains',
            'food_supply_relevance', 'image_quality', 'uncertainties'}


class DirectHTTPClient:
    """Minimal Messages API transport using Python HTTPS, bypassing SDK transport."""
    def __init__(self):
        self.messages=self
    def create(self, **payload):
        req=urllib.request.Request('https://api.anthropic.com/v1/messages',
            data=json.dumps(payload).encode(),headers={
                'content-type':'application/json',
                'x-api-key':os.environ['ANTHROPIC_API_KEY'],
                'anthropic-version':'2026-10-21'},method='POST')
        with urllib.request.urlopen(req,timeout=45) as response:
            body=json.load(response)
            request_id=response.headers.get('request-id')
        return SimpleNamespace(content=[SimpleNamespace(**b) for b in body['content']],
            stop_reason=body.get('stop_reason'),_request_id=request_id,
            usage=SimpleNamespace(model_dump=lambda:body.get('usage',{})))
    def close(self):
        pass


def prepare(frame):
    h, w = frame.shape[:2]
    scale = min(1., 1024/w, 768/h)
    image = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return image, data.tobytes()


def quality(frame):
    small = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark = float((gray < 8).mean())
    bright = float((gray > 247).mean())
    return dict(sharpness=sharpness, dark_fraction=dark, bright_fraction=bright)


def parse_result(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict) or not REQUIRED.issubset(result):
        raise ValueError('Missing required observation fields')
    for name in ['hazard_signs', 'people', 'curtains', 'uncertainties']:
        if not isinstance(result[name], list):
            raise ValueError(f'{name} must be a list')
    for name in ['scene', 'stairs', 'food_supply_relevance']:
        if not isinstance(result[name], dict):
            raise ValueError(f'{name} must be an object')
    if type(result['stairs'].get('observed')) is not bool:
        raise ValueError('stairs.observed must be boolean')
    return result


class Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = deque(maxlen=8)
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        try:
            if not cap.isOpened():
                self.error = 'Cannot open direct RTSP stream'
                return
            last_sample = 0
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = 'Camera stream interrupted; restart client'
                    return
                now = time.monotonic()
                if now-last_sample < .1:
                    continue
                last_sample = now
                utc = datetime.now(timezone.utc).isoformat()
                with self.lock:
                    self.frames.append((now, utc, frame))
        except Exception as exc:
            self.error = str(exc)
        finally:
            cap.release()

    def latest(self):
        with self.lock:
            return self.frames[-1] if self.frames else None

    def select(self):
        with self.lock:
            samples = list(self.frames)
        now = time.monotonic()
        candidates = []
        for received, utc, frame in samples:
            if now-received > 1.:
                continue
            q = quality(frame)
            # Conservative gross-exposure check, not a readability guarantee.
            if q['dark_fraction'] > .85 or q['bright_fraction'] > .85:
                continue
            candidates.append((q['sharpness'], received, utc, frame, q))
        return max(candidates, key=lambda x: x[0]) if candidates else None


def analyse(client, model, prompt, candidate, folder, dry_run):
    _, received, utc, frame, q = candidate
    image, jpeg = prepare(frame)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = folder/stamp
    path.mkdir()
    (path/'submitted.jpg').write_bytes(jpeg)
    record = dict(received_at_utc=utc, timestamp_source='local RTSP frame arrival, not sensor capture time',
        selected_age_s=round(time.monotonic()-received, 3), image_width=image.shape[1],
        image_height=image.shape[0], quality=q, model=model, dry_run=dry_run,
        robot_pose=None, success=False)
    started = time.monotonic()
    try:
        if dry_run:
            record.update(success=True, result=None)
        else:
            response = client.messages.create(model=model, max_tokens=1800,
                system=prompt, messages=[{'role':'user','content':[
                    {'type':'image','source':{'type':'base64','media_type':'image/jpeg',
                        'data':base64.b64encode(jpeg).decode('ascii')}},
                    {'type':'text','text':'Report observations for this image using the required JSON format.'}]}])
            text = '\n'.join(b.text for b in response.content if b.type == 'text')
            record.update(raw_text=text, request_id=getattr(response, '_request_id', None),
                stop_reason=response.stop_reason, usage=response.usage.model_dump())
            if response.stop_reason == 'max_tokens':
                raise ValueError('Response truncated; observation not accepted')
            record.update(result=parse_result(text), success=True)
    except urllib.error.HTTPError as exc:
        try:
            body=json.loads(exc.read().decode())
            detail=body.get('error',{})
        except Exception:
            detail={}
        message=str(detail.get('message','Server rejected request'))
        key=os.environ.get('ANTHROPIC_API_KEY','')
        if key:message=message.replace(key,'[REDACTED]')
        message=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',message)
        record.update(error=f'Claude HTTP {exc.code}', status_code=exc.code,
            api_error_type=detail.get('type'), api_error_message=message[:1200])
    except urllib.error.URLError as exc:
        record.update(error_type=type(exc).__name__,connection_causes=[type(exc.reason).__name__],
            error='HTTPS connection failed')
    except anthropic.APIConnectionError as exc:
        causes=[]
        cause=exc.__cause__
        while cause is not None:
            causes.append(type(cause).__name__)
            cause=cause.__cause__
        record.update(error_type=type(exc).__name__, connection_causes=causes,
            error='Could not complete HTTPS connection to Claude; no observation accepted')
    except anthropic.APIStatusError as exc:
        body=getattr(exc,'body',None)
        details=body.get('error',body) if isinstance(body,dict) else {}
        message=str(details.get('message','No error message provided')) if isinstance(details,dict) else 'No error message provided'
        key=os.environ.get('ANTHROPIC_API_KEY','')
        if key:message=message.replace(key,'[REDACTED]')
        message=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',message)
        record.update(error=f'Claude HTTP {exc.status_code}', status_code=exc.status_code,
            api_error_type=details.get('type') if isinstance(details,dict) else None,
            api_error_message=message[:1200], request_id=getattr(exc,'request_id',None))
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error='Request or result validation failed; no observation accepted')
    record['elapsed_s'] = round(time.monotonic()-started, 3)
    record['files'] = str(path)
    (path/'result.json').write_text(json.dumps(record, indent=2))
    (folder/'latest.json').write_text(json.dumps(record, indent=2))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=os.environ.get('CLAUDE_MODEL','claude-opus-5-5'))
    p.add_argument('--interval', type=float, default=5.)
    p.add_argument('--max-requests', type=int, default=3, help='0 for unlimited; default bounded first test')
    p.add_argument('--seconds', type=float, default=0)
    p.add_argument('--dry-run', action='store_true', help='Capture and prepare locally; no API calls')
    p.add_argument('--headless', action='store_true')
    args = p.parse_args()
    if args.interval <= 0 or args.max_requests < 0 or args.seconds < 0:
        p.error('Invalid interval or limits')
    if not args.dry_run and not os.environ.get('ANTHROPIC_API_KEY'):
        p.error('ANTHROPIC_API_KEY is missing. Set it locally; never paste it into chat.')
    if not args.dry_run:
        key=os.environ['ANTHROPIC_API_KEY'].strip()
        if not key.isascii() or any(c.isspace() for c in key):
            p.error('API key contains whitespace or non-ASCII characters. Copy only the key and retry.')
        os.environ['ANTHROPIC_API_KEY']=key
    prompt = (ROOT/'mission_prompt.txt').read_text()
    client = None if args.dry_run else DirectHTTPClient()
    folder = ROOT/'vlm_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True)
    (folder/'mission_prompt.txt').write_text(prompt)
    camera = Camera()
    pool = ThreadPoolExecutor(max_workers=1)
    future = None
    count = 0
    next_due = time.monotonic()+2
    start = time.monotonic()
    status = 'Waiting for camera'
    print('Mode:', 'LOCAL DRY RUN' if args.dry_run else f'CLAUDE API: {args.model}',flush=True)
    print('Results:',folder,flush=True)
    print('Q to quit. Each request is independent; observations do not control the robot.',flush=True)
    try:
        while not args.seconds or time.monotonic()-start < args.seconds:
            if camera.error:
                print(camera.error,flush=True)
                break
            now = time.monotonic()
            if future is not None and future.done():
                result = future.result()
                status = 'Prepared locally' if args.dry_run else ('Observation received' if result['success'] else 'Request failed')
                print(json.dumps(result,ensure_ascii=False),flush=True)
                future = None
                if result.get('status_code') in (400,401,403,404):
                    print('Stopping: check API key, model access, or request configuration.',flush=True)
                    break
                if args.max_requests and count >= args.max_requests:
                    break
            if now >= next_due:
                next_due = now+args.interval
                if future is None:
                    candidate = camera.select()
                    if candidate is not None:
                        future = pool.submit(analyse,client,args.model,prompt,candidate,folder,args.dry_run)
                        count += 1
                        status = 'Preparing locally' if args.dry_run else 'Claude request in progress'
                    else:
                        status = 'No recent usable frame'
            latest = camera.latest()
            if not args.headless and latest:
                preview = cv2.resize(latest[2],(960,540))
                cv2.putText(preview,status,(15,30),0,.65,(0,255,0),2)
                cv2.putText(preview,f'Requests: {count} | interval: {args.interval}s | Q quit',(15,60),0,.6,(0,255,0),2)
                cv2.imshow('Periodic scene understanding - group8',preview)
                if cv2.waitKey(1)&255 == ord('q'):
                    break
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop.set()
        if future is not None:
            print('Waiting for current request to finish (timeout 45s); no new requests.',flush=True)
        pool.shutdown(wait=True)
        camera.thread.join(timeout=6)
        if not args.headless:
            cv2.destroyAllWindows()
        if client:
            client.close()


if __name__ == '__main__':
    main()
