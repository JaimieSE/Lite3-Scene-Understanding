#!/usr/bin/env python3
"""Read-only direct RTSP viewer; all files and processing stay on group8."""
import argparse
import os
import threading
import time
from datetime import datetime
from pathlib import Path
import cv2
import zxingcpp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--decoder', choices=['opencv','zxing','both'], default='opencv')
    p.add_argument('--seconds',type=float,default=0)
    args=p.parse_args()
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;tcp'
    cap=cv2.VideoCapture('rtsp://192.168.1.120:8554/test',cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000])
    if not cap.isOpened():raise SystemExit('Could not open RTSP stream')
    lock=threading.Lock();stop=threading.Event()
    state={'frame':None,'seq':0,'received':0,'error':None}
    def capture():
        while not stop.is_set():
            ok,frame=cap.read()
            if not ok:
                with lock:state['error']='Stream read failed; restart viewer to reconnect'
                break
            with lock:
                state['frame']=frame;state['seq']+=1;state['received']=time.monotonic()
    worker=threading.Thread(target=capture,daemon=True);worker.start()
    start=time.monotonic();last_seq=-1;processed=0;last_decode=0;codes=[];previous=set()
    folder=Path(__file__).resolve().parent/'direct_captures'
    folder.mkdir(exist_ok=True)
    print('Direct RTSP viewer. Q: quit | S: save full-resolution PNG. No ROS dependency.',flush=True)
    try:
        while not stop.is_set() and (not args.seconds or time.monotonic()-start<args.seconds):
            with lock:
                frame=state['frame'];seq=state['seq'];received=state['received'];error=state['error']
            if error:print(error,flush=True);break
            if frame is None or seq==last_seq:
                if not args.headless and cv2.waitKey(1)&255==ord('q'):break
                time.sleep(.005);continue
            last_seq=seq;processed+=1
            if time.monotonic()-last_decode>.2:
                codes=[]
                if args.decoder in ('opencv','both'):
                    detector=cv2.QRCodeDetector()
                    text,points,_=detector.detectAndDecode(frame)
                    if points is not None:
                        codes.append(dict(text=text,points=points.reshape(4,2),source='OpenCV'))
                    ok,texts,polygons,_=detector.detectAndDecodeMulti(frame)
                    if polygons is not None:
                        for index,polygon in enumerate(polygons):
                            value=texts[index] if index<len(texts) else ''
                            if value and any(c['text']==value for c in codes):continue
                            codes.append(dict(text=value,points=polygon.reshape(4,2),source='OpenCV'))
                if args.decoder in ('zxing','both'):
                    for b in zxingcpp.read_barcodes(frame,formats=zxingcpp.BarcodeFormat.QRCode):
                        if any(c['text']==b.text for c in codes):continue
                        pts=[b.position.top_left,b.position.top_right,b.position.bottom_right,b.position.bottom_left]
                        codes.append(dict(text=b.text,points=[[v.x,v.y] for v in pts],source='ZXing'))
                last_decode=time.monotonic()
                current={b['text'] for b in codes if b['text']}
                for text in current-previous:print('QR:',repr(text),flush=True)
                previous=current
            preview=frame.copy()
            for b in codes:
                import numpy as np
                polygon=np.array(b['points'],dtype='int32')
                cv2.polylines(preview,[polygon],True,(0,255,0),3)
                label=b['text'][:80] or 'QR detected - NOT decoded'
                cv2.putText(preview,label,(max(0,int(polygon[0,0])),max(30,int(polygon[0,1])-10)),0,.7,(0,255,0),2)
            cv2.putText(preview,f"Direct RTSP | {args.decoder} | Decoded: {sum(bool(c['text']) for c in codes)} | Q quit, S save",(15,30),0,.65,(0,255,0),2)
            if not args.headless:
                cv2.imshow('Direct Lite3 camera - group8',cv2.resize(preview,(960,540)))
                key=cv2.waitKey(1)&255
                if key==ord('q'):break
                if key==ord('s'):
                    path=folder/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.png')
                    cv2.imwrite(str(path),frame);print('Saved:',path,flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set();worker.join(timeout=6)
        if not worker.is_alive():cap.release()
        if not args.headless:cv2.destroyAllWindows()
        with lock:count=state['seq'];frame=state['frame']
        if args.headless and frame is not None:
            path=folder/'latest_test.png';cv2.imwrite(str(path),frame);print('Saved:',path,flush=True)
        print(f'Captured {count} frames; processed {processed}; elapsed {time.monotonic()-start:.1f}s. Initial capture can include buffered frames.',flush=True)


if __name__=='__main__':main()
