# multi person detection using yolo
# mediapipe only detects 1 person so we use yolov8-pose here
# it finds all people + their skeleton keypoints in one shot

import cv2
from ultralytics import YOLO

# load the nano model - smallest and fastest one, good for raspi
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
print("Press q to quit")

while True:
    ok, frame = cap.read()
    if not ok: break

    # run yolo on the frame
    # results[0].keypoints.xy gives us (N, 17, 2) array
    # N = number of people detected
    # 17 = COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
    # 2 = x, y pixel coordinates
    results = model(frame, verbose=False, conf=0.5)

    # yolo has built-in drawing, this draws everything
    annotated = results[0].plot()

    # but you can also get the raw keypoints like this:
    if results[0].keypoints is not None:
        kps = results[0].keypoints.xy.cpu().numpy()
        num_people = kps.shape[0]
        cv2.putText(annotated, f"People: {num_people}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # loop through each person detected
        for i in range(num_people):
            person_kps = kps[i]  # shape = (17, 2)
            # keypoint indices:
            # 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
            # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow
            # 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
            # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

            # hip center - good point for tracking where someone is standing
            hx = (person_kps[11][0] + person_kps[12][0]) / 2
            hy = (person_kps[11][1] + person_kps[12][1]) / 2
            cv2.circle(annotated, (int(hx), int(hy)), 6, (0, 0, 255), -1)
            cv2.putText(annotated, f"P{i}", (int(hx)+10, int(hy)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Multi Person Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
