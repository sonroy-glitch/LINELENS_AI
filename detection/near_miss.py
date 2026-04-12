# near miss detection - checks if 2 people get too close to each other
# uses yolov8 to find all people then measures distances between them
# if distance < threshold = NEAR MISS = safety event

import cv2, math, time, json
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# safety distance in pixels
# to figure out the right value: stand 1 meter apart from someone
# and see how many pixels that is in your camera feed
SAFE_DISTANCE_PX = 150
near_misses = []

print("Near-miss detection running. Press q to quit.")

while True:
    ok, frame = cap.read()
    if not ok: break
    ts = time.time()

    results = model(frame, verbose=False, conf=0.5)

    if results[0].keypoints is not None:
        kps = results[0].keypoints.xy.cpu().numpy()
        num_people = kps.shape[0]

        # get hip center for each person (index 11=left_hip, 12=right_hip)
        centers = []
        for i in range(num_people):
            hx = (kps[i][11][0] + kps[i][12][0]) / 2
            hy = (kps[i][11][1] + kps[i][12][1]) / 2
            centers.append((int(hx), int(hy)))
            cv2.circle(frame, (int(hx), int(hy)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"P{i}", (int(hx)+8, int(hy)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # check every pair of people
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = math.hypot(centers[i][0]-centers[j][0],
                                  centers[i][1]-centers[j][1])

                if dist < SAFE_DISTANCE_PX:
                    # too close! draw red line
                    cv2.line(frame, centers[i], centers[j], (0, 0, 255), 3)
                    mid_x = (centers[i][0] + centers[j][0]) // 2
                    mid_y = (centers[i][1] + centers[j][1]) // 2
                    cv2.putText(frame, f"NEAR MISS! {dist:.0f}px",
                                (mid_x-60, mid_y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    near_misses.append({
                        "ts": ts, "person_a": i, "person_b": j,
                        "distance_px": round(dist, 1)
                    })
                else:
                    # safe distance - green line
                    cv2.line(frame, centers[i], centers[j], (0, 255, 0), 1)

        cv2.putText(frame, f"People: {num_people} | Near misses: {len(near_misses)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Near Miss Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

with open("near_misses.json", "w") as f:
    json.dump(near_misses, f, indent=2)
print(f"Total near misses: {len(near_misses)}")
