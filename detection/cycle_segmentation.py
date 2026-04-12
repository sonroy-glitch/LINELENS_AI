import cv2
import mediapipe as mp
import numpy as np
import json
import time

with open("zones_config.json", "r") as f:
    raw_cfg = json.load(f)


polys = {
    k: np.array(v, dtype=np.int32)
    for k, v in raw_cfg.items()
}


cfg = {
    "start_zone": "safe_zone",
    "end_zone": "safe_zone",
    "min_cycle_s": 3.0
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)


def point_in(poly, x, y):
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


current_cycle_start = None
cycle_id = 0
cycles = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    ts = time.time()

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        
        hx = (lm[23].x + lm[24].x) * w / 2.0
        hy = (lm[23].y + lm[24].y) * h / 2.0

       
        in_start = point_in(polys[cfg["start_zone"]], hx, hy)
        in_end   = point_in(polys[cfg["end_zone"]], hx, hy)

        in_hazard = point_in(polys["hazard_zone"], hx, hy)
        in_machine = point_in(polys["machines"], hx, hy)

     
        if in_hazard:
            cv2.putText(frame, "HAZARD!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            print(" Worker in HAZARD zone!")

        if in_machine:
            cv2.putText(frame, "Near Machine", (50,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    
        if current_cycle_start is None:
            if in_start:
                current_cycle_start = ts
                print(f"Cycle started: {cycle_id}")

        #time.time()
        else:
            dur = ts - current_cycle_start

            if in_end and dur >= cfg["min_cycle_s"]:
                cycles.append({
                    "cycle_id": cycle_id,
                    "start": current_cycle_start,
                    "end": ts,
                    "dur": dur
                })

                print(f"Cycle ended: {cycle_id} | Duration: {round(dur,2)}s")

                cycle_id += 1
                current_cycle_start = None

        
        cv2.circle(frame, (int(hx), int(hy)), 5, (0,255,255), -1)

    
    for name, poly in polys.items():
        color = (255,255,255)

        if name == "safe_zone":
            color = (0,255,0)
        elif name == "hazard_zone":
            color = (0,0,255)
        elif name == "machines":
            color = (255,0,0)

        cv2.polylines(frame, [poly], True, color, 2)
        cv2.putText(frame, name, tuple(poly[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

  
    cv2.imshow("Smart Workcell Observer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        with open('cycles.json','w') as f:
            json.dump(cycles,f)
        break

cap.release()
cv2.destroyAllWindows()

print("\nRecorded cycles:", len(cycles))

for c in cycles:
    print(c)