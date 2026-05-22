# sop compliance checker
# you define the correct order of zones a worker should visit
# it watches what they actually do and compares
# outputs a "drift %" showing how much they deviated

import cv2, mediapipe as mp, numpy as np, json, time

# load your zone polygons
with open("../config/zones_config.json", "r") as f:
    zones = json.load(f)
polys = {k: np.array(v, dtype=np.int32) for k, v in zones.items()}


# the correct order the worker should visit zones
# change these to match your zones_config.json zone names
EXPECTED_SEQUENCE = ["safe_zone", "hazard_zone", "machines", "safe_zone"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

def point_in(poly, x, y):
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

# track which zones the worker visited in order
visited_zones = []
last_zone = None

print("SOP compliance checker running. Press q to quit.")
print(f"Expected sequence: {' -> '.join(EXPECTED_SEQUENCE)}")

while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        # hip center
        hx = (lm[23].x + lm[24].x) * w / 2
        hy = (lm[23].y + lm[24].y) * h / 2
        cv2.circle(frame, (int(hx), int(hy)), 5, (0, 255, 255), -1)

        # which zone are they in?
        current_zone = None
        for name, poly in polys.items():
            if point_in(poly, hx, hy):
                current_zone = name
                break

        # if they entered a new zone, log it
        if current_zone and current_zone != last_zone:
            visited_zones.append(current_zone)
            last_zone = current_zone
            print(f"  Step {len(visited_zones)}: entered {current_zone}")

        if current_zone:
            cv2.putText(frame, f"In: {current_zone}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # calculate drift %
    # count how many expected steps were done in the right order
    matched = 0
    exp_idx = 0
    for v in visited_zones:
        if exp_idx < len(EXPECTED_SEQUENCE) and v == EXPECTED_SEQUENCE[exp_idx]:
            matched += 1
            exp_idx += 1

    drift = (1 - matched / max(len(EXPECTED_SEQUENCE), 1)) * 100

    # color code: green if good, orange if drifting
    drift_color = (0, 165, 255) if drift > 20 else (0, 255, 0)
    cv2.putText(frame, f"SOP Drift: {drift:.0f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, drift_color, 2)
    cv2.putText(frame, f"Steps: {len(visited_zones)}/{len(EXPECTED_SEQUENCE)}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # draw zones on frame
    for name, poly in polys.items():
        cv2.polylines(frame, [poly], True, (255, 255, 255), 1)
        cv2.putText(frame, name, tuple(poly[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("SOP Compliance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

result = {
    "expected": EXPECTED_SEQUENCE,
    "actual": visited_zones,
    "drift_pct": round(drift, 1),
    "matched_steps": matched
}
with open("sop_result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nDrift: {drift:.1f}%")
print(f"Expected: {' -> '.join(EXPECTED_SEQUENCE)}")
print(f"Actual:   {' -> '.join(visited_zones)}")
