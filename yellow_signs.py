"""Colour/shape proposals only; these are not confirmed hazards."""
import cv2
import numpy as np


def detect_yellow_triangles(frame, min_area=300):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([15,80,65]),np.array([40,255,255]))
    kernel=np.ones((3,3),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        area=cv2.contourArea(c)
        if area<min_area or area>frame.shape[0]*frame.shape[1]*.5:continue
        perimeter=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,.045*perimeter,True)
        if len(approx)!=3 or not cv2.isContourConvex(approx):continue
        x,y,w,h=cv2.boundingRect(c)
        if not .45<w/h<2.:continue
        hull_area=cv2.contourArea(cv2.convexHull(c))
        if hull_area<=0 or area/hull_area<.8:continue
        region=np.zeros(mask.shape,np.uint8);cv2.drawContours(region,[c],-1,255,-1)
        fill=cv2.countNonZero(cv2.bitwise_and(mask,region))/max(1,cv2.countNonZero(region))
        if fill<.3:continue
        candidates.append(dict(box=[x,y,x+w,y+h],triangle=approx.reshape(-1,2).tolist(),
            yellow_fraction=round(fill,3),area=round(area,1)))
    return sorted(candidates,key=lambda v:v['area'],reverse=True)[:8]


def crop_sign(frame, candidate):
    x1,y1,x2,y2=candidate['box'];pad=max(8,int(max(x2-x1,y2-y1)*.15))
    return frame[max(0,y1-pad):min(frame.shape[0],y2+pad),
                 max(0,x1-pad):min(frame.shape[1],x2+pad)].copy()


def fingerprint(crop):
    gray=cv2.cvtColor(cv2.resize(crop,(9,8)),cv2.COLOR_BGR2GRAY)
    return (gray[:,1:]>gray[:,:-1]).flatten()


def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    intersection=max(0,x2-x)*max(0,y2-y)
    return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


class SignTracker:
    def __init__(self):self.tracks=[];self.next_id=1
    def update(self, candidates, now):
        old=[t for t in self.tracks if now-t['last']<1.]
        used=set();tracks=[]
        for c in candidates:
            matches=[(iou(c['box'],t['box']),i,t) for i,t in enumerate(old) if i not in used]
            match=max(matches,key=lambda v:v[0]) if matches else None
            if match and match[0]>.4:
                _,i,t=match;used.add(i)
                track=dict(c,id=t['id'],hits=t['hits']+1,last=now)
            else:
                track=dict(c,id=self.next_id,hits=1,last=now);self.next_id+=1
            tracks.append(track)
        self.tracks=tracks
        return tracks
