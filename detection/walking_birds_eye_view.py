import cv2, mediapipe as mp, math, time, json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

prev = None
walk_px = 0.0
px_per_m = 500# a hyperparameter 

positions = []  

while True:
    ok, frame = cap.read()
    if not ok: break
    h,w = frame.shape[:2]
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ts = time.time()
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        # print(lm[23])
        if lm[7].visibility>0.6 and lm[8].visibility>0.6:
            x = (lm[7].x + lm[8].x) * w / 2.0
            y = (lm[7].y + lm[8].y) * h / 2.0
            positions.append((ts,x,y))
            if prev is not None:
                dx = abs(x - prev[0]); dy = abs(y - prev[1])
                walk_px += math.hypot(dx, dy)
            prev = (x,y)
            cv2.circle(frame, (int(x),int(y)), 3, (255,255,255), -1)
    cv2.putText(frame, f"walk_m: {walk_px/px_per_m:.2f} m", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
    cv2.imshow("Walking", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows()
with open("walk_positions.json","w") as f:
    json.dump({"walk_px":walk_px, "walk_m":walk_px/px_per_m}, f)
print("Saved walk_m:", walk_px/px_per_m)