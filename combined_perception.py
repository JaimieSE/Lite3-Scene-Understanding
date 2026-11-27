#!/usr/bin/env python3
"""OpenCV direct RTSP capture + local YOLO inference. No ROS or robot commands."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import cv2
import torch
import numpy as np
import zxingcpp
from ultralytics import YOLO
from periodic_vlm import Camera
from yellow_signs import detect_yellow_triangles, SignTracker, crop_sign, fingerprint
from sign_vlm import identify

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(ROOT/'yolo26n.pt'))
    p.add_argument('--confidence',type=float,default=.4)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--vlm',action='store_true',help='Enable paid Claude sign identification')
    p.add_argument('--dry-run',action='store_true',help='Save sign triggers locally; no API calls')
    p.add_argument('--vlm-model',default=os.environ.get('CLAUDE_MODEL','claude-opus-5-5'))
    p.add_argument('--max-requests',type=int,default=3)
    p.add_argument('--cooldown',type=float,default=30)
    p.add_argument('--min-area',type=int,default=300)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seconds',type=float,default=0)
    args=p.parse_args()
    if not 0<args.confidence<=1 or args.threads<1 or args.imgsz<32 or args.seconds<0:
        p.error('Invalid confidence, thread count, image size or duration')
    if not Path(args.model).expanduser().is_file():
        p.error('Model file not found; supply an existing .pt file')
    if args.max_requests<0 or args.cooldown<0 or args.min_area<1:
        p.error('Invalid trigger limits')
    if args.vlm and not args.dry_run:
        key=os.environ.get('ANTHROPIC_API_KEY','').strip()
        if not key or not key.isascii() or any(c.isspace() for c in key):
            p.error('Set ANTHROPIC_API_KEY locally or use the launcher hidden prompt')
        os.environ['ANTHROPIC_API_KEY']=key
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    model=YOLO(str(Path(args.model).expanduser()))
    print('Classes:',json.dumps(model.names),flush=True)
    print('One direct RTSP capture -> YOLO + QR + yellow triangles + optional triggered VLM. Q quit | S save evidence.',flush=True)
    print('Pretrained weights do not include curtain/staircase. Use --model for custom weights.',flush=True)
    camera=Camera()
    stop=threading.Event();lock=threading.Lock()
    state={'result':None,'count':0,'error':None}
    folder=ROOT/'perception_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');folder.mkdir(parents=True)
    tracker=SignTracker()
    pool=ThreadPoolExecutor(max_workers=1);future=None;seen=[];last_sent=-1e9
    sent=0;handled=None;labels={};vlm_status='VLM off'
    print('Results:',folder,flush=True)
    print('VLM mode:', 'local trigger dry-run' if args.dry_run else 'enabled' if args.vlm else 'off',flush=True)
    def infer():
        last=None
        try:
            while not stop.is_set():
                sample=camera.latest()
                if sample is None or sample[0]==last:
                    stop.wait(.005);continue
                received,utc,frame=sample;last=received
                if time.monotonic()-received>1:
                    stop.wait(.02);continue
                started=time.monotonic()
                qr_codes=[]
                qr_text,qr_points,_=cv2.QRCodeDetector().detectAndDecode(frame)
                if qr_points is not None:
                    qr_codes.append(dict(text=qr_text,decoded=bool(qr_text),source='OpenCV',corners=qr_points.reshape(4,2).tolist()))
                for barcode in zxingcpp.read_barcodes(frame,formats=zxingcpp.BarcodeFormat.QRCode):
                    if any(q['text']==barcode.text for q in qr_codes):continue
                    pts=[barcode.position.top_left,barcode.position.top_right,barcode.position.bottom_right,barcode.position.bottom_left]
                    qr_codes.append(dict(text=barcode.text,decoded=True,source='ZXing',corners=[[v.x,v.y] for v in pts]))
                result=model.predict(frame,imgsz=args.imgsz,conf=args.confidence,device='cpu',verbose=False)[0]
                boxes=[dict(label=result.names[int(b.cls.item())],confidence=float(b.conf.item()),
                    xyxy=[float(v) for v in b.xyxy[0].tolist()]) for b in result.boxes]
                yellow_signs=tracker.update(detect_yellow_triangles(frame,args.min_area),time.monotonic())
                record=dict(received_at_utc=utc,timestamp_source='local RTSP arrival',
                    frame_received=received,processing_ms=round((time.monotonic()-started)*1000,1),
                    image_width=frame.shape[1],image_height=frame.shape[0],objects=boxes,qr_codes=qr_codes,yellow_sign_candidates=yellow_signs)
                with lock:
                    state['result']=(frame,record);state['count']+=1
        except Exception as exc:
            with lock:state['error']=f'{type(exc).__name__}: {exc}'
    worker=threading.Thread(target=infer,daemon=True);worker.start()
    started=time.monotonic();last_print=started;last_count=0;fps=0.;last_frame=None
    try:
        while not args.seconds or time.monotonic()-started<args.seconds:
            if camera.error:
                print('Camera:',camera.error,flush=True);break
            with lock:
                output=state['result'];count=state['count'];error=state['error']
            if error:raise RuntimeError(error)
            now=time.monotonic()
            if future is not None and future.done():
                response=future.result();future=None
                print('VLM result:',json.dumps(response),flush=True)
                for entry in response.get('result',{}).get('signs',[]):labels[entry['candidate_id']]=entry
                vlm_status='Identification received' if response['success'] and not args.dry_run else 'Trigger saved locally' if args.dry_run else 'VLM failed; see terminal'
                if response.get('status_code') in (400,401,403,404):args.vlm=False
            if output and handled!=output[1]['frame_received']:
                handled=output[1]['frame_received']
                seen=[(when,f) for when,f in seen if now-when<args.cooldown]
                eligible=[]
                for sign in output[1]['yellow_sign_candidates']:
                    if sign['hits']<3:continue
                    crop=crop_sign(output[0],sign)
                    if min(crop.shape[:2])<40:continue
                    f=fingerprint(crop)
                    if any(np.count_nonzero(f!=other)<=8 for _,other in seen):continue
                    eligible.append((sign,f))
                permitted=not args.max_requests or sent<args.max_requests
                fresh=now-output[1]['frame_received']<1
                if eligible and future is None and permitted and fresh and now-last_sent>=5 and (args.vlm or args.dry_run):
                    chosen=eligible[:4]
                    future=pool.submit(identify,output[0].copy(),output[1]['received_at_utc'],[v[0] for v in chosen],args.vlm_model,folder,args.dry_run)
                    seen.extend((now,f) for _,f in chosen);last_sent=now;sent+=1
                    vlm_status='Identifying signs...' if not args.dry_run else 'Saving trigger...'
                    print('VLM trigger:',[v[0]['id'] for v in chosen],flush=True)
            if args.max_requests and sent>=args.max_requests and future is None:
                vlm_status=f'Request limit reached: {sent}; local detection continues'
            if now-last_print>=2:
                fps=(count-last_count)/(now-last_print)
                if output:
                    record=output[1]
                    print(json.dumps(dict(inference_fps=round(fps,1),processing_ms=record['processing_ms'],
                        objects=record['objects'],qr_codes=record['qr_codes'])),flush=True)
                last_print=now;last_count=count
            sample=camera.latest()
            if sample is None:
                time.sleep(.01);continue
            frame=(output[0] if output else sample[2]).copy()
            # Overlay only results from a recent frame; never queue stale frames.
            if output:
                for b in output[1]['objects']:
                    x1,y1,x2,y2=map(int,b['xyxy'])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"{b['label']} {b['confidence']:.2f}",
                        (x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
                for sign in output[1].get('yellow_sign_candidates',[]):
                    sx1,sy1,sx2,sy2=sign['box']
                    cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),(0,255,255),2)
                    label=f"#{sign['id']} possible warning sign"
                    entry=labels.get(sign['id'])
                    if entry:
                        label=f"#{sign['id']} "+(entry['hazard_label'] if entry['is_warning_sign'] else 'Not a warning sign')+(' ?' if entry['uncertain'] else '')
                    cv2.putText(frame,label,(sx1,max(75,sy1-8)),0,.6,(0,255,255),2)
                for qr in output[1]['qr_codes']:
                    polygon=np.asarray(qr['corners'],dtype='int32')
                    cv2.polylines(frame,[polygon],True,(255,255,0),3)
                    label=('QR: '+qr['text'][:70]) if qr['decoded'] else 'QR located - not decoded'
                    cv2.putText(frame,label,(max(0,int(polygon[0,0])),max(50,int(polygon[0,1])-10)),0,.65,(255,255,0),2)
            cv2.putText(frame,f'YOLO {fps:.1f} inference FPS | Q quit | S save',
                (15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
            cv2.putText(frame,f'{vlm_status} | sign requests {sent}',(15,58),0,.6,(0,255,255),2)
            last_frame=frame
            if not args.headless:
                cv2.imshow('Lite3 combined perception',cv2.resize(frame,(960,540)))
                key=cv2.waitKey(1)&255
                if key==ord('q'):break
                if key==ord('s') and output:
                    path=folder/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                    path.mkdir()
                    # Save the exact inference frame, not a newer preview frame.
                    cv2.imwrite(str(path/'frame.png'),output[0])
                    record=dict(output[1]);record.pop('frame_received',None);record['sign_identifications']=dict(labels)
                    cv2.imwrite(str(path/'annotated.jpg'),frame)
                    (path/'detections.json').write_text(json.dumps(record,indent=2))
                    print('Saved:',path,flush=True)
            time.sleep(.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set();worker.join(timeout=15)
        camera.stop.set();camera.thread.join(timeout=6)
        if future is not None:print('Finishing current API request; no new requests.',flush=True)
        pool.shutdown(wait=True)
        if not args.headless:cv2.destroyAllWindows()
        with lock:output=state['result'];count=state['count']
        if args.headless and output:
            cv2.imwrite(str(folder/'latest_test.png'),output[0])
            record=dict(output[1]);record.pop('frame_received',None);record['sign_identifications']=dict(labels)
            (folder/'latest_test.json').write_text(json.dumps(record,indent=2))
        print(f'Finished: {count} inference frames in {time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':main()
