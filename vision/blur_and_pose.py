
import cv2
import mediapipe as mp
# print(dir(mp))
mp_pose = mp.solutions.pose
face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
#0,1,2
k=51
cap = cv2.VideoCapture(0)
# cap1 = cv2.VideoCapture(1)
# cap2 = cv2.VideoCapture(2)

print("Press q to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h,w = frame.shape[:2]
    # cvtColor(frame/image,what_conversion_you_are_gonna_do)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res1 = pose.process(rgb)
    res2=face.process(rgb)
    out = frame.copy()
    # what does any drawing utils expect(frame/image,pose_landmarks or the points on the frame for
    #out major joints,)
    # pose _detection_model
    if res1.pose_landmarks:
        mp_drawing.draw_landmarks(out, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # print(res1.pose_landmarks)
    #face_blurring_technique
    if res2.detections:
        h,w = frame.shape[:2]
        for d in res2.detections:
            box = d.location_data.relative_bounding_box
            # print(box)
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin+box.width)*w))
            y2 = min(h, int((box.ymin+box.height)*h))
            if x2>x1 and y2>y1:
                roi = out[y1:y2, x1:x2]
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k,k), 0)
    cv2.imshow("Pose Demo", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()