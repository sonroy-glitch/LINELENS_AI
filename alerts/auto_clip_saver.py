# auto clip saver - automatically saves video clips when events happen
# unlike events_detection.py where you press 'e' manually,
# this one triggers automatically when it detects something
# (hazard zone entry, bad posture, near miss etc)

import cv2, mediapipe as mp, numpy as np, time, os, json
from collections import deque

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

# how many seconds before and after the event to save
PRE_ROLL = 4.0
POST_ROLL = 6.0

# frame buffer - stores recent frames so we can go "back in time"
buf = deque()

# load zones for hazard detection
try:
    with open("../config/zones_config.json") as f:
        zones = json.load(f)
    polys = {k: np.array(v, dtype=np.int32) for k, v in zones.items()}
    has_zones = True
except:
    has_zones = False
    polys = {}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

out_dir = "../clips"
os.makedirs(out_dir, exist_ok=True)

def point_in(poly, x, y):
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

# cooldown so we dont save 100 clips per second
last_save_ts = 0
COOLDOWN = 10.0  # seconds between clip saves

recording = False
record_start = 0
record_frames = []

print("Auto clip saver running. Clips save automatically on events.")
print("Press q to quit.")

while True:
    ok, frame = cap.read()
    if not ok: break
    ts = time.time()

    # add to rolling buffer
    buf.append((ts, frame.copy()))
    # trim old frames
    while buf and (ts - buf[0][0]) > (PRE_ROLL + POST_ROLL + 2):
        buf.popleft()

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    event_detected = None

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        hx = (lm[23].x + lm[24].x) * w / 2
        hy = (lm[23].y + lm[24].y) * h / 2

        # check hazard zone
        if has_zones and "hazard_zone" in polys:
            if point_in(polys["hazard_zone"], hx, hy):
                event_detected = "hazard_zone"
                cv2.putText(frame, "HAZARD ZONE!", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # if recording post-roll frames
    if recording:
        record_frames.append((ts, frame.copy()))
        if ts - record_start > POST_ROLL:
            # save the clip now
            if len(record_frames) > 5:
                fname = os.path.join(out_dir, f"auto_{event_detected}_{int(ts)}.mp4")
                vw = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
                for _, fr in record_frames:
                    vw.write(fr)
                vw.release()
                print(f"  Saved clip: {fname} ({len(record_frames)} frames)")
            recording = False
            record_frames = []

    # trigger new recording
    if event_detected and not recording and (ts - last_save_ts) > COOLDOWN:
        recording = True
        record_start = ts
        last_save_ts = ts
        # grab pre-roll frames from buffer
        pre_cutoff = ts - PRE_ROLL
        record_frames = [(t, f.copy()) for (t, f) in buf if t >= pre_cutoff]
        print(f"  Event: {event_detected} - recording clip...")

    # show status
    status = "REC" if recording else "monitoring"
    cv2.putText(frame, f"[{status}] clips saved: check /clips folder",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Auto Clip Saver", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
