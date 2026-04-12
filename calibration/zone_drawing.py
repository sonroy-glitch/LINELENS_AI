import cv2
import json
import os
#((0,0),(2,3),(3,4),(4,5))
import numpy as np

cfg_path = "zones_config.json"

canvas_w = 640
canvas_h = 360

# img = 255 * np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)
img=cv2.imread("factory_layout1.jpg")
img=cv2.resize(img,(canvas_w,canvas_h))
polys = {}
current = []

def mouse_cb(event, x, y, flags, param):
    global current

    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x, y))


cv2.namedWindow("Zone Drawer")
cv2.setMouseCallback("Zone Drawer", mouse_cb)

print("Instructions:")
print("- Click to place polygon points")
print("- Press 'n' to name and save polygon")
print("- Press 'r' to reset current polygon")
print("- Press 's' to save zones to file")
print("- Press 'q' to quit")

while True:

    disp = img.copy()
    #safe-green , unsafe - red
   
    for name, poly in polys.items():
        # pts=[]
        pts = np.array(poly, dtype=np.int32)

        cv2.polylines(disp, [pts], True, (0,0,255), 2)

        cv2.putText(
            disp,
            name,
            tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

    
    if len(current) > 0:

        pts = np.array(current, dtype=np.int32)

        cv2.polylines(disp, [pts], False, (0,255,0), 1)

        for pt in current:
            cv2.circle(disp, pt, 4, (0,255,0), -1)

    cv2.imshow("Zone Drawer", disp)

    key = cv2.waitKey(20) & 0xFF

    
    if key == ord('n'):

        if len(current) < 3:
            print("Polygon needs at least 3 points")
            continue

        name = input("Polygon name: ").strip()

        if name != "":
            polys[name] = list(current)
            current = []

    
    elif key == ord('r'):

        current = []

   
    elif key == ord('s'):

        with open(cfg_path, "w") as f:
            json.dump(polys, f, indent=2)
            
        print("Saved zones to", cfg_path)

   
    elif key == ord('q'):

        break


cv2.destroyAllWindows()