#!/usr/bin/env python3
"""OpenCV direct RTSP capture + local YOLO inference. No ROS or robot commands."""
import argparse
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
from yellow_signs import detect_yellow_triangles

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(ROOT/'yolo26n.pt'))
    p.add_argument('--confidence',type=float,default=.4)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seconds',type=float,default=0)
    args=p.parse_args()
    if not 0<args.confidence<=1 or args.threads<1 or args.imgsz<32 or args.seconds<0:
        p.error('Invalid confidence, thread count, image size or duration')
    if not Path(args.model).expanduser().is_file():
        p.error('Model file not found; supply an existing .pt file')
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    model=YOLO(str(Path(args.model).expanduser()))
    print('Classes:',json.dumps(model.names),flush=True)
    print('Direct RTSP -> OpenCV + YOLO + QR decoding. Q quit | S save evidence.',flush=True)
    print('Pretrained weights do not include curtain/staircase. Use --model for custom weights.',flush=True)
    camera=Camera()
    stop=threading.Event();lock=threading.Lock()
    state={'result':None,'count':0,'error':None}
    folder=ROOT/'yolo_captures';folder.mkdir(exist_ok=True)
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
                yellow_signs=detect_yellow_triangles(frame)
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
            frame=sample[2].copy()
            # Overlay only results from a recent frame; never queue stale frames.
            if output and now-output[1]['frame_received']<.4:
                for b in output[1]['objects']:
                    x1,y1,x2,y2=map(int,b['xyxy'])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"{b['label']} {b['confidence']:.2f}",
                        (x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
                for sign in output[1].get('yellow_sign_candidates',[]):
                    sx1,sy1,sx2,sy2=sign['box']
                    cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),(0,255,255),2)
                    cv2.putText(frame,'Possible warning sign',(sx1,max(25,sy1-8)),0,.6,(0,255,255),2)
                for qr in output[1]['qr_codes']:
                    polygon=np.asarray(qr['corners'],dtype='int32')
                    cv2.polylines(frame,[polygon],True,(255,255,0),3)
                    label=('QR: '+qr['text'][:70]) if qr['decoded'] else 'QR located - not decoded'
                    cv2.putText(frame,label,(max(0,int(polygon[0,0])),max(50,int(polygon[0,1])-10)),0,.65,(255,255,0),2)
            cv2.putText(frame,f'YOLO {fps:.1f} inference FPS | Q quit | S save',
                (15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
            last_frame=frame
            if not args.headless:
                cv2.imshow('Lite3 OpenCV + YOLO - direct RTSP',cv2.resize(frame,(960,540)))
                key=cv2.waitKey(1)&255
                if key==ord('q'):break
                if key==ord('s') and output:
                    path=folder/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                    path.mkdir()
                    # Save the exact inference frame, not a newer preview frame.
                    cv2.imwrite(str(path/'frame.png'),output[0])
                    record=dict(output[1]);record.pop('frame_received',None)
                    (path/'detections.json').write_text(json.dumps(record,indent=2))
                    print('Saved:',path,flush=True)
            time.sleep(.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set();worker.join(timeout=15)
        camera.stop.set();camera.thread.join(timeout=6)
        if not args.headless:cv2.destroyAllWindows()
        with lock:output=state['result'];count=state['count']
        if args.headless and output:
            cv2.imwrite(str(folder/'latest_test.png'),output[0])
            record=dict(output[1]);record.pop('frame_received',None)
            (folder/'latest_test.json').write_text(json.dumps(record,indent=2))
        print(f'Finished: {count} inference frames in {time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':main()
