# skeleton only video - maximum privacy mode
# instead of saving actual camera footage,
# this only saves the skeleton (stick figure) on a black background
# no faces, no clothes, no identity - just the pose data

import cv2, mediapipe as mp, numpy as np, time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# output video writer for skeleton-only footage
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
w_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
h_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
out = cv2.VideoWriter("skeleton_only.mp4",
                       cv2.VideoWriter_fourcc(*"mp4v"),
                       fps, (w_out, h_out))

print("Recording skeleton-only video. Press q to stop.")
print("This is the privacy-safe version - no faces stored!")

while True:
    ok, frame = cap.read()
    if not ok: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    # create blank black frame - this is what we save
    # no camera image at all, just the skeleton
    black = np.zeros_like(frame)

    if res.pose_landmarks:
        # draw skeleton on the black frame
        mp_drawing.draw_landmarks(black, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # also draw on camera frame for live preview
        mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # write the skeleton-only frame to file
    out.write(black)

    # show both side by side
    # left = what the camera sees, right = what gets saved
    combined = np.hstack([frame, black])
    combined = cv2.resize(combined, (1280, 480))
    cv2.putText(combined, "Camera (NOT saved)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(combined, "Skeleton Only (SAVED)", (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Privacy Mode", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved skeleton_only.mp4 - no identifiable information stored!")
