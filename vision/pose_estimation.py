
import cv2
import mediapipe as mp
# print(dir(mp))
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
#0,1,2
cap = cv2.VideoCapture(0)
# cap1 = cv2.VideoCapture(1)
# cap2 = cv2.VideoCapture(2)

print("Press q to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h,w = frame.shape[:2]
    # print(h,w)
    # cvtColor(frame/image,what_conversion_you_are_gonna_do)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    # what does any drawing utils expect(frame/image,pose_landmarks or the points on the frame for
    #out major joints,)
    if res.pose_landmarks:
        mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        print(res.pose_landmarks)
    cv2.imshow("Pose Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()