import cv2, mediapipe as mp, time, math
import json
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

last_hip = None
idle_start = None
idle_total = 0.0
motion_thresh = 3.0  # px per frame threshold
min_idle_s = 1.5
idles=[]
id=0
while True:
    ok, frame = cap.read()
    if not ok: break
    h,w = frame.shape[:2]
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ts = time.time()
    hip = None
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        if lm[23].visibility>0.73 and lm[24].visibility>0.73:
            x = (lm[23].x + lm[24].x) * w / 2.0
            y = (lm[23].y + lm[24].y) * h / 2.0
            hip = (x,y)
            cv2.circle(frame, (int(x),int(y)), 3, (255,255,255), -1)
    if hip is not None and last_hip is not None:
        d = math.hypot(hip[0]-last_hip[0], hip[1]-last_hip[1])
    else:
        d = 0.0
    last_hip = hip
    if d < motion_thresh:
        if idle_start is None:
            idle_start = ts
    else:
        if idle_start is not None:
            chunk = ts - idle_start
            if chunk >= min_idle_s:
                idle_total += chunk
                idles.append({'id':id,'start_time':idle_start,'end_time':ts})
                id+=1
            idle_start = None

    cv2.putText(frame, f"idle_s_total: {idle_total:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
    cv2.imshow("Idle Detection", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows()
with open("idle.json","w") as f:
    json.dump(idles, f)

print("Idle seconds recorded:", idle_total)