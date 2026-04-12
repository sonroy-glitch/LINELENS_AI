# perspective calibration - convert pixels to real world meters
# click 2 points on a known distance (like a 1 meter tile)
# and it calculates px_per_meter for you
# then all your walking distances will be in real meters

import cv2, math, json

cap = cv2.VideoCapture(0)
print("Press SPACE to capture a frame for calibration...")

# grab a frame
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("Press SPACE to capture", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cap.release()
cv2.destroyAllWindows()

if frame is None:
    print("Error: no frame captured")
    exit()

# click 2 points
points = []
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print(f"  Point {len(points)}: ({x}, {y})")

cv2.namedWindow("Click 2 points on a known distance")
cv2.setMouseCallback("Click 2 points on a known distance", click)

print("\nClick TWO points that span a known real-world distance")
print("(like the corners of a 1 meter floor tile)")

while len(points) < 2:
    disp = frame.copy()
    for pt in points:
        cv2.circle(disp, pt, 6, (0, 0, 255), -1)
    if len(points) == 2:
        cv2.line(disp, points[0], points[1], (0, 255, 0), 2)
    cv2.imshow("Click 2 points on a known distance", disp)
    cv2.waitKey(50)

# show the line
disp = frame.copy()
cv2.line(disp, points[0], points[1], (0, 255, 0), 2)
cv2.imshow("Click 2 points on a known distance", disp)
cv2.waitKey(500)
cv2.destroyAllWindows()

# calculate pixel distance
px_dist = math.hypot(points[1][0]-points[0][0], points[1][1]-points[0][1])
print(f"\nPixel distance: {px_dist:.1f} px")

# ask for real distance
real_m = float(input("Enter the real distance in METERS: "))
px_per_m = px_dist / max(real_m, 0.01)

print(f"\npx_per_meter = {px_per_m:.1f}")
print(f"(use this value in walking.py and other scripts)")

# save it
data = {"px_per_meter": round(px_per_m, 2), "points_px": points, "real_meters": real_m}
with open("calibration.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved to calibration.json")
