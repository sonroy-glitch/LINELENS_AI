# ergonomics detection - finds unsafe body postures
# checks: bending too much, reaching overhead, deep squats
# uses joint angles from mediapipe pose landmarks

import cv2, mediapipe as mp, math, time, json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# thresholds - tweak these based on your factory/workstation
BEND_ANGLE = 45       # trunk angle below this = bad bend (in degrees)
SQUAT_ANGLE = 90      # knee angle below this = deep squat
OVERHEAD_TIME = 2.0   # wrist above shoulder for this many seconds = warning

violations = []
overhead_start = None  # timestamp when overhead reach started
violation_id = 0

def angle(a, b, c):
    # calculates the angle at point b formed by lines a-b and b-c
    # a, b, c are (x, y) tuples
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba < 0.001 or mag_bc < 0.001:
        return 180.0
    cos_a = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_a))

print("Ergonomics monitor running. Press q to quit.")

while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    ts = time.time()

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # helper to get pixel coords from landmark index
        def pt(idx):
            return (lm[idx].x * w, lm[idx].y * h)

        # joints we need
        # using visibility as a proxy for confidence from mediapipe
        shoulder_l = (pt(11), lm[11].visibility)
        shoulder_r = (pt(12), lm[12].visibility)
        hip_l = (pt(23), lm[23].visibility)
        hip_r = (pt(24), lm[24].visibility)
        knee_l = (pt(25), lm[25].visibility)
        knee_r = (pt(26), lm[26].visibility)
        ankle_l = (pt(27), lm[27].visibility)
        ankle_r = (pt(28), lm[28].visibility)
        wrist_l = (pt(15), lm[15].visibility)
        wrist_r = (pt(16), lm[16].visibility)

        # --- 1. TRUNK BEND CHECK (Relative to vertical gravity) ---
        s_best = shoulder_l[0] if shoulder_l[1] > shoulder_r[1] else shoulder_r[0]
        h_best = hip_l[0] if hip_l[1] > hip_r[1] else hip_r[0]
        
        vec_y = h_best[1] - s_best[1]
        vec_x = abs(h_best[0] - s_best[0])
        trunk_angle = math.degrees(math.atan2(vec_x, max(vec_y, 0.001)))
        
        if trunk_angle > BEND_ANGLE:
            cv2.putText(frame, f"BAD BEND! {trunk_angle:.0f}deg", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            violations.append({"id": violation_id, "type": "bend",
                                "angle": round(trunk_angle, 1), "ts": ts})
            violation_id += 1

        # --- 2. SQUAT CHECK ---
        conf_l = hip_l[1] + knee_l[1] + ankle_l[1]
        conf_r = hip_r[1] + knee_r[1] + ankle_r[1]
        
        knee_angle = 180
        if conf_l > conf_r and knee_l[1] > 0.5:
            knee_angle = angle(hip_l[0], knee_l[0], ankle_l[0])
        elif knee_r[1] > 0.5:
            knee_angle = angle(hip_r[0], knee_r[0], ankle_r[0])
        
        if knee_angle < SQUAT_ANGLE:
            cv2.putText(frame, f"DEEP SQUAT! {knee_angle:.0f}deg", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            violations.append({"id": violation_id, "type": "squat",
                                "angle": round(knee_angle, 1), "ts": ts})
            violation_id += 1

        # --- 3. OVERHEAD REACH CHECK ---
        overhead = False
        if wrist_l[1] > 0.5 and shoulder_l[1] > 0.5 and wrist_l[0][1] < shoulder_l[0][1]:
            overhead = True
        if wrist_r[1] > 0.5 and shoulder_r[1] > 0.5 and wrist_r[0][1] < shoulder_r[0][1]:
            overhead = True
            
        if overhead:
            if overhead_start is None:
                overhead_start = ts
            dur = ts - overhead_start
            if dur >= OVERHEAD_TIME:
                cv2.putText(frame, f"OVERHEAD REACH! {dur:.1f}s", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                violations.append({"id": violation_id, "type": "overhead",
                                    "duration": round(dur, 1), "ts": ts})
                violation_id += 1
        else:
            overhead_start = None

        # show live angles at bottom of screen
        cv2.putText(frame, f"trunk: {trunk_angle:.0f} knee: {knee_angle:.0f}",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Violations: {len(violations)}", (w-200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Ergonomics Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# save all violations to json
with open("ergo_violations.json", "w") as f:
    json.dump(violations, f, indent=2)
print(f"Saved {len(violations)} violations to ergo_violations.json")
