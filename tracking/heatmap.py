# heatmap - shows where people spend the most time
# accumulates position data and draws a color overlay
# red = lots of time spent there, blue = rarely visited

import cv2, mediapipe as mp, numpy as np, time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# grid resolution - lower = faster, higher = more detail
GRID_W = 64
GRID_H = 48
grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)

print("Heatmap recording. Press q to quit and save.")

while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        if lm[23].visibility > 0.5 and lm[24].visibility > 0.5:
            # hip center
            x = (lm[23].x + lm[24].x) / 2
            y = (lm[23].y + lm[24].y) / 2

            # convert to grid coordinates
            gx = int(x * GRID_W)
            gy = int(y * GRID_H)
            gx = max(0, min(GRID_W-1, gx))
            gy = max(0, min(GRID_H-1, gy))

            # add to the accumulator
            grid[gy, gx] += 1.0

    # render the heatmap overlay
    # normalize grid to 0-255
    if grid.max() > 0:
        norm = (grid / grid.max() * 255).astype(np.uint8)
    else:
        norm = grid.astype(np.uint8)

    # apply jet colormap (blue=cold, red=hot)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (w, h))

    # blend with camera frame
    # 0.6 = how much camera you see, 0.4 = how much heatmap
    blended = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    cv2.imshow("Occupancy Heatmap", blended)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# save the final heatmap as an image
if grid.max() > 0:
    norm = (grid / grid.max() * 255).astype(np.uint8)
    final_heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    final_heatmap = cv2.resize(final_heatmap, (640, 480))
    cv2.imwrite("heatmap.png", final_heatmap)
    print("Saved heatmap.png")
