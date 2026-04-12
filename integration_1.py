
import cv2, json, time, numpy as np, mediapipe as mp, math
from shapely.geometry import Point, Polygon 
import os 

zones = json.load(open("zones_config.json"))


polys = {k: Polygon(v) for k,v in zones.items()}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1)
face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

def hip_center(kps):
    return ((kps['l_hip']['x']+kps['r_hip']['x'])/2.0, (kps['l_hip']['y']+kps['r_hip']['y'])/2.0)

events = []

while True:
    ok, frame = cap.read()
    if not ok: break
    h,w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_pose = pose.process(rgb)
    res_face = face.process(rgb)
    hip = None
    if res_pose.pose_landmarks:
        lm = res_pose.pose_landmarks.landmark
        kps = {"l_hip":{"x":lm[23].x*w,"y":lm[23].y*h}, "r_hip":{"x":lm[24].x*w,"y":lm[24].y*h}}
        hx, hy = hip_center(kps)
        #()-tuple
        #[]-list
        hip = (int(hx), int(hy))
        cv2.circle(frame, hip, 4, (255,255,255), -1)
        for name, poly in polys.items():
            inside = poly.contains(Point(hx, hy))
            if inside:
                cv2.putText(frame, f"In {name}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
                events.append({"ts": time.time(), "type": "hip_in_"+name})
    cv2.imshow("Zone Events", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): 
        with open('events.json','w') as f:
            json.dump(events,f)
        break

cap.release(); cv2.destroyAllWindows()
print("Collected events:", len(events))