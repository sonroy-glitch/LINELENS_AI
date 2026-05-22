# queue / wait time detection
# detects when a person is standing still for too long
# in a real multi-person setup youd check if multiple people
# are idle in the same area = bottleneck

import cv2, mediapipe as mp, time, math, json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# if person moves less than this many pixels = "standing still"
MOTION_THRESH = 4.0
# need to be still for this many seconds to count as "waiting"
WAIT_TIME = 3.0

last_pos = None
idle_start = None
queue_events = []

print("Queue/wait detection running. Press q to quit.")

while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    ts = time.time()
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        # hip center for position tracking
        if lm[23].visibility > 0.5 and lm[24].visibility > 0.5:
            x = (lm[23].x + lm[24].x) * w / 2
            y = (lm[23].y + lm[24].y) * h / 2
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)

            # check how much they moved since last frame
            if last_pos:
                d = math.hypot(x - last_pos[0], y - last_pos[1])
                if d < MOTION_THRESH:
                    # barely moved = waiting
                    if idle_start is None:
                        idle_start = ts
                    wait_dur = ts - idle_start
                    if wait_dur > WAIT_TIME:
                        cv2.putText(frame, f"WAITING: {wait_dur:.1f}s", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        # draw a circle showing the wait zone
                        cv2.circle(frame, (int(x), int(y)), 40, (0, 165, 255), 2)
                else:
                    # they moved - end the wait period
                    if idle_start and (ts - idle_start) > WAIT_TIME:
                        queue_events.append({
                            "start": idle_start,
                            "end": ts,
                            "duration": round(ts - idle_start, 1),
                            "x": round(x, 1),
                            "y": round(y, 1)
                        })
                        print(f"  Wait event: {ts - idle_start:.1f}s")
                    idle_start = None
            last_pos = (x, y)

    cv2.putText(frame, f"Wait events: {len(queue_events)}", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Queue Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

with open("queue_events.json", "w") as f:
    json.dump(queue_events, f, indent=2)
print(f"Queue events recorded: {len(queue_events)}")
