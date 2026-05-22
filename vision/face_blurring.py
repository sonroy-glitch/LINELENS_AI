import cv2, mediapipe as mp

face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
k = 51  # this is called blur ratio

while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face.process(rgb) # coordinates for the bounding boxes 
    out = frame.copy()
    if res.detections:
        h,w = frame.shape[:2]#(height,width,channels)
        for d in res.detections:
            box = d.location_data.relative_bounding_box
            print(box)
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin+box.width)*w))
            y2 = min(h, int((box.ymin+box.height)*h))
            if x2>x1 and y2>y1:
                roi = out[y1:y2, x1:x2]
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k,k), 0)
    cv2.imshow("Face Blur", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()