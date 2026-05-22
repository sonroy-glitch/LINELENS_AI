
import cv2, mediapipe as mp, time, os
from collections import deque

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
before_s = 4.0
after_s = 8.0
buf = deque()  # (ts, frame)
#[],(),{1:"2"}
out_dir = "clips"
os.makedirs(out_dir, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

print("Press 'e' to simulate an event and save a clip. q to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break
    ts = time.time()
    buf.append((ts, frame.copy()))
    # trim
    while buf and (ts - buf[0][0]) > (before_s + after_s + 2.0):
        buf.popleft()

    cv2.imshow("Clip Buffer", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('e'):
        # write clip
        t0 = ts - before_s
        t1 = ts + after_s
        frames = [f for (t,f) in buf if t0 <= t <= t1]
        if len(frames) >= 5:
            fname = os.path.join(out_dir, f"event_{int(ts)}.mp4")
            vw = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
            for fr in frames:
                vw.write(fr)
            vw.release()
            print("Saved clip", fname)
        else:
            print("Not enough frames for clip")
    elif k == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()