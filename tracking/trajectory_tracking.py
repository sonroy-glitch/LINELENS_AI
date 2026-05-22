import cv2
import mediapipe as mp
import numpy as np
import json
from collections import deque
import time

with open("zones_config.json") as f:
    zones = json.load(f)

polys = {k: np.array(v, dtype=np.int32) for k,v in zones.items()}


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)


trajectory = deque(maxlen=100)
changes=[]
def point_in(poly, x,y):
    return cv2.pointPolygonTest(poly, (float(x),float(y)), False) >= 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h,w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    tm=time.time()
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        hx = int((lm[7].x + lm[8].x) * w / 2)
        hy = int((lm[7].y + lm[8].y) * h / 2)

        trajectory.append((hx,hy))

        in_hazard = point_in(polys["assembly"], hx, hy)
        in_machine = point_in(polys["machine area"], hx, hy)

     
        if in_hazard:
            if len(changes)!=0:
                val=changes[-1]
                if not val.get("Hazard",None):
                    changes.append({"Hazard":tm})
            else:
                changes.append({"Hazard":tm})
            cv2.putText(frame, "HAZARD!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            print(" Worker in HAZARD zone!")

        if in_machine:
            if len(changes)!=0:
                val=changes[-1]
                if not val.get("Machine",None):
                    changes.append({"Machine":tm})
            else:
                changes.append({"Machine":tm})
            cv2.putText(frame, "Near Machine", (50,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (0,255,255), 2)

        cv2.circle(frame, (hx,hy), 5, (0,0,255), -1)
        for name, poly in polys.items():
            color = (255,255,255)

            if name == "office" or name == "warehouse" or name == "shipping area"  :
                color = (0,255,0)
            elif name == "assembly" or name == "raw materials":
                color = (0,0,255)
            elif name == "machine 1" or name == "machine 2" or name == "machine 3" or name=="machine area":
                color = (255,0,0)

            cv2.polylines(frame, [poly], True, color, 2)
            cv2.putText(frame, name, tuple(poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    for name, poly in polys.items():
        cv2.polylines(frame, [poly], True, (255,255,255), 1)

    cv2.imshow("Trajectory", frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        with open("changes.json", "w") as f:
            json.dump(changes, f)
        break

cap.release()
cv2.destroyAllWindows()