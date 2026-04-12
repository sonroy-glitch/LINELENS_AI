# spaghetti diagram - classic IE tool
# draws ALL walking paths on the factory layout image
# shows you where people walk the most and where theres waste

import cv2, mediapipe as mp, numpy as np, math, json, time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# try to load factory layout as background
# if you dont have one it just uses a dark canvas
try:
    layout = cv2.imread("../config/factory_layout1.jpg")
    layout = cv2.resize(layout, (640, 480))
except:
    layout = np.ones((480, 640, 3), dtype=np.uint8) * 30

# store all positions for the full session
all_positions = []  # list of (x, y) tuples
total_walk_px = 0
prev = None
px_per_m = 500  # same hyperparameter as walking.py

print("Spaghetti diagram recording. Walk around then press q.")

while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        if lm[23].visibility > 0.5 and lm[24].visibility > 0.5:
            x = (lm[23].x + lm[24].x) * w / 2
            y = (lm[23].y + lm[24].y) * h / 2
            all_positions.append((int(x), int(y)))

            if prev:
                total_walk_px += math.hypot(x-prev[0], y-prev[1])
            prev = (x, y)

    # draw live trail on camera frame
    for i in range(1, len(all_positions)):
        cv2.line(frame, all_positions[i-1], all_positions[i], (0, 255, 255), 1)

    cv2.putText(frame, f"Walk: {total_walk_px/px_per_m:.2f}m | Points: {len(all_positions)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Recording Path", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# now draw the spaghetti diagram on the layout image
diagram = layout.copy()

# draw the full path in yellow
for i in range(1, len(all_positions)):
    cv2.line(diagram, all_positions[i-1], all_positions[i], (0, 0, 255), 1, cv2.LINE_AA)

# put total distance annotation
walk_m = total_walk_px / px_per_m
cv2.putText(diagram, f"Total walk: {walk_m:.1f}m", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(diagram, "Spaghetti Diagram", (200, 470),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

cv2.imwrite("spaghetti_diagram.png", diagram)
print(f"Saved spaghetti_diagram.png (total walk: {walk_m:.1f}m)")

# show it
cv2.imshow("Spaghetti Diagram", diagram)
cv2.waitKey(0)
cv2.destroyAllWindows()
