# run everything together - the full LineLens AI pipeline
# uses YOLOv8-pose to detect MULTIPLE people and runs all the logic on them
# tracking idle time, walk distance, ergonomics, zones, sop drift, cycles, near misses
# also triggers LEDs and saves video clips on events!
# press q to quit, it generates a shift report at the end

import cv2, numpy as np, math, time, json, os, threading, subprocess, sys
import mediapipe as mp
from collections import deque
from ultralytics import YOLO

# import our LED script
try:
    from alerts import led_buzzer
    HAS_LEDS = True
except ImportError:
    HAS_LEDS = False
    print("Could not load LED buzzer script")

mp_face = mp.solutions.face_detection
face_det = mp_face.FaceDetection(min_detection_confidence=0.4)

# load the yolo model for multi-person detection
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture('./test_videos/worker-zone-detection.mp4')
fps_cap = cap.get(cv2.CAP_PROP_FPS) or 15.0
w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

# try to load zones
try:
    with open("config/zones_config.json") as f:
        zones = json.load(f)
    polys = {k: np.array(v, dtype=np.int32) for k, v in zones.items()}
    has_zones = True
    print(f"Loaded {len(polys)} zones")
except:
    has_zones = False
    polys = {}
    print("No zones loaded (run calibration/zone_drawing.py first)")

# --- hyperparameters ---
IDLE_THRESH = 4.0     # pixels of motion below this = idle
IDLE_TIME = 3.0       # seconds before flagging as idle
PX_PER_M = 500        # pixels per meter (from calibration)
BEND_ANGLE = 45       # trunk angle for bad bend
SQUAT_ANGLE = 90      # knee angle for deep squat
OVERHEAD_TIME = 2.0   # seconds wrist is above shoulder
SAFE_DISTANCE = 150   # pixels for near miss detection
EXPECTED_SOP = ["safe_zone", "hazard_zone", "machines", "safe_zone"]

# --- global state ---
events_log = []
violations = []
near_misses = []
queue_events = []
completed_cycles = []
all_walk_paths = []
frame_count = 0
session_start = time.time()

# auto clip saver buffer
video_buffer = deque(maxlen=int(fps_cap * 10)) # 10 seconds rolling buffer
recordings = []

class PersonState:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = (x, y)
        self.trail = deque(maxlen=100)
        
        # idle tracking
        self.idle_start = None
        self.total_idle = 0
        self.total_walk = 0
        
        # zone tracking
        self.zone = None
        self.zone_history = []
        self.cycle_start = time.time()
        
        # ergo tracking
        self.overhead_start = None

# simple centroid tracking
active_persons = {}
next_person_id = 0

# heatmap grid
GRID_W, GRID_H = 64, 48
heatgrid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
show_heatmap = False

def angle_at(a, b, c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    m1 = math.hypot(ba[0], ba[1])
    m2 = math.hypot(bc[0], bc[1])
    if m1 < 0.001 or m2 < 0.001: return 180
    return math.degrees(math.acos(max(-1, min(1, dot/(m1*m2)))))

def point_in(poly, x, y):
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def fire_alert(alert_type):
    # runs alert in background thread so it doesnt freeze video
    if HAS_LEDS:
        if alert_type == "danger": threading.Thread(target=led_buzzer.alert_danger, daemon=True).start()
        elif alert_type == "warning": threading.Thread(target=led_buzzer.alert_warning, daemon=True).start()
        elif alert_type == "ergo": threading.Thread(target=led_buzzer.alert_ergo, daemon=True).start()

def save_clip_async(event_name, frames_list):
    # saves video clip in background
    def write_video():
        os.makedirs("clips", exist_ok=True)
        fname = f"clips/event_{event_name}_{int(time.time())}.mp4"
        vw = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"mp4v"), fps_cap, (w_cap, h_cap))
        for fr in frames_list: vw.write(fr)
        vw.release()
        print(f"\n[Clip Saved] {fname}")
    threading.Thread(target=write_video, daemon=True).start()

print("\n====== LineLens AI Running ======")
print("  MULTI-PERSON YOLO + FULL LOGIC")
print("  q = quit and save reports")
print("  h = toggle heatmap")
print("=================================\n")

fps_timer = time.time()
fps = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (640, 480))
    ts = time.time()

    # keep frame for clip saving
    video_buffer.append(frame.copy())
    
    # --- face blur (privacy) ---
    # apply heavy gaussian blur to any detected faces before YOLO draws over them
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_res = face_det.process(rgb)
    if face_res.detections:
        for det in face_res.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (51, 51), 30)

    # run yolo
    results = model(frame, verbose=False, conf=0.5)
    
    # render the joint connections and bounding boxes directly from YOLO
    frame = results[0].plot()
    
    current_centroids = []
    current_kps = []
    
    if results[0].keypoints is not None:
        # Use .data to get [x, y, confidence] for filtering out noise
        kps = results[0].keypoints.data.cpu().numpy()
        for i in range(kps.shape[0]):
            pkp = kps[i] # (17, 3) geometry
            # hip center (11=left_hip, 12=right_hip)
            l_hip, r_hip = pkp[11], pkp[12]
            
            hx, hy, count = 0, 0, 0
            if l_hip[2] > 0.5:
                hx += l_hip[0]; hy += l_hip[1]; count += 1
            if r_hip[2] > 0.5:
                hx += r_hip[0]; hy += r_hip[1]; count += 1
                
            if count > 0:
                hx /= count; hy /= count
                current_centroids.append((hx, hy))
                current_kps.append(pkp)

                # add to heatmap
                gx = int(hx / w * GRID_W); gy = int(hy / h * GRID_H)
                gx = max(0, min(GRID_W-1, gx)); gy = max(0, min(GRID_H-1, gy))
                heatgrid[gy, gx] += 1

    # --- simple tracker ---
    matched_ids = []
    for (cx, cy) in current_centroids:
        best_id = None
        best_dist = 9999
        for pid, p in active_persons.items():
            if pid in matched_ids: continue
            dist = math.hypot(cx - p.pos[0], cy - p.pos[1])
            if dist < 100 and dist < best_dist:
                best_dist = dist
                best_id = pid
                
        if best_id is not None:
            active_persons[best_id].pos = (cx, cy)
            active_persons[best_id].trail.append((int(cx), int(cy)))
            matched_ids.append(best_id)
        else:
            p = PersonState(next_person_id, cx, cy)
            active_persons[next_person_id] = p
            matched_ids.append(next_person_id)
            next_person_id += 1

    active_persons = {pid: p for pid, p in active_persons.items() if pid in matched_ids}
    
    person_zones = {}
    triggered_clip_this_frame = None

    # --- per-person logic ---
    for i, pid in enumerate(matched_ids):
        p = active_persons[pid]
        pkp = current_kps[i]
        
        # 1. idle and walking
        if len(p.trail) >= 2:
            last = p.trail[-2]
            curr = p.trail[-1]
            all_walk_paths.append((last, curr))
            dist_moved = math.hypot(curr[0]-last[0], curr[1]-last[1])
            p.total_walk += dist_moved
            
            if dist_moved < IDLE_THRESH:
                if p.idle_start is None: p.idle_start = ts
                dur = ts - p.idle_start
                if dur > IDLE_TIME:
                    cv2.putText(frame, f"P{pid} IDLE: {dur:.1f}s", (int(p.pos[0]), int(p.pos[1])-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                if p.idle_start and (ts - p.idle_start) > IDLE_TIME:
                    idle_dur = ts - p.idle_start
                    p.total_idle += idle_dur
                    events_log.append({"type": "idle", "pid": pid, "dur": round(idle_dur,1), "ts": ts})
                    fire_alert("warning")
                p.idle_start = None

        # 2. zones & SOP & cycles
        if has_zones:
            current_z = None
            for name, poly in polys.items():
                if point_in(poly, p.pos[0], p.pos[1]):
                    current_z = name
                    break
            
            person_zones[pid] = current_z

            if current_z != p.zone:
                if current_z:
                    p.zone_history.append(current_z)
                    events_log.append({"type": "zone_entry", "pid": pid, "zone": current_z, "ts": ts})
                    
                    if "hazard" in current_z.lower():
                        fire_alert("danger")
                        triggered_clip_this_frame = f"hazard_p{pid}"

                    # cycle detection: if they returned to the first zone in the SOP
                    if len(EXPECTED_SOP) > 0 and current_z == EXPECTED_SOP[0] and len(p.zone_history) > 1:
                        dur = ts - p.cycle_start
                        if dur > 10.0:  # ignore super short loops
                            # calculate SOP drift for this cycle
                            matched = 0
                            e_idx = 0
                            for z in p.zone_history:
                                if e_idx < len(EXPECTED_SOP) and z == EXPECTED_SOP[e_idx]:
                                    matched += 1
                                    e_idx += 1
                            drift = (1.0 - (matched / max(len(EXPECTED_SOP), 1))) * 100
                            
                            completed_cycles.append({
                                "pid": pid, "dur": round(dur,1), "drift_pct": round(drift,1), "ts": ts
                            })
                            print(f"Cycle finished! Dur: {dur:.1f}s | Drift: {drift:.1f}%")
                            
                            p.zone_history = [current_z]
                            p.cycle_start = ts
                p.zone = current_z
            
            if p.zone:
                color = (0,0,255) if "hazard" in p.zone.lower() else (0,255,0)
                cv2.putText(frame, p.zone, (int(p.pos[0]), int(p.pos[1])+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 3. ERGONOMICS (Robust logic with confidence filtering)
        # Using [x, y, conf] geometry
        sl = pkp[5]; sr = pkp[6]; hl = pkp[11]; hr = pkp[12]
        kl = pkp[13]; kr = pkp[14]; al = pkp[15]; ar = pkp[16]
        wl = pkp[9]; wr = pkp[10]

        bad_posture = False

        # --- TRUNK BEND (Angle relative to vertical gravity) ---
        # Get best shoulder and best hip based on confidence
        s_best = sl if sl[2] > sr[2] else sr
        h_best = hl if hl[2] > hr[2] else hr
        
        if s_best[2] > 0.5 and h_best[2] > 0.5:
            # y goes down, so h_best[1] - s_best[1] is positive if shoulder is above hip
            vec_y = h_best[1] - s_best[1] 
            vec_x = abs(h_best[0] - s_best[0])
            # angle from vertical (0 = standing straight, 90 = bending horizontal)
            trunk_bend = math.degrees(math.atan2(vec_x, max(vec_y, 0.001)))
            
            if trunk_bend > BEND_ANGLE: # Now checking > BEND_ANGLE instead of <
                cv2.putText(frame, f"P{pid} BEND!", (int(p.pos[0]), int(p.pos[1])-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                violations.append({"type": "bend", "pid": pid, "angle": round(trunk_bend,1), "ts": ts})
                bad_posture = True

        # --- SQUAT (Knee angle) ---
        # use the leg that the AI is most confident about
        conf_l = hl[2] + kl[2] + al[2]
        conf_r = hr[2] + kr[2] + ar[2]
        
        knee_a = 180
        if conf_l > conf_r and kl[2] > 0.5:
            knee_a = angle_at(hl, kl, al)
        elif kr[2] > 0.5:
            knee_a = angle_at(hr, kr, ar)

        if knee_a < SQUAT_ANGLE:
            cv2.putText(frame, f"P{pid} SQUAT!", (int(p.pos[0]), int(p.pos[1])-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            violations.append({"type": "squat", "pid": pid, "angle": round(knee_a,1), "ts": ts})
            bad_posture = True

        # --- OVERHEAD REACH ---
        overhead = False
        if wl[2] > 0.5 and sl[2] > 0.5 and wl[1] < sl[1]: overhead = True
        if wr[2] > 0.5 and sr[2] > 0.5 and wr[1] < sr[1]: overhead = True
        
        if overhead:
            if p.overhead_start is None: p.overhead_start = ts
            dur = ts - p.overhead_start
            if dur > OVERHEAD_TIME:
                cv2.putText(frame, f"P{pid} OVERHEAD!", (int(p.pos[0]), int(p.pos[1])-90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                violations.append({"type": "overhead", "pid": pid, "dur": round(dur,1), "ts": ts})
                bad_posture = True
        else:
            p.overhead_start = None

        if bad_posture and len(violations) % 10 == 1: # throttle alerts
            fire_alert("ergo")
            triggered_clip_this_frame = f"ergo_p{pid}"

        # draw trail
        for t_idx in range(1, len(p.trail)):
            cv2.line(frame, p.trail[t_idx-1], p.trail[t_idx], (255, 255, 0), 1)

    # --- multi-person logic ---
    centroids = [active_persons[pid].pos for pid in matched_ids]
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            d = math.hypot(centroids[i][0]-centroids[j][0], centroids[i][1]-centroids[j][1])
            if d < SAFE_DISTANCE:
                cv2.line(frame, (int(centroids[i][0]), int(centroids[i][1])), 
                                (int(centroids[j][0]), int(centroids[j][1])), (0,0,255), 2)
                near_misses.append({"type": "near_miss", "dist": round(d,1), "ts": ts})
                if len(near_misses) % 5 == 1:  # throttle
                    fire_alert("danger")
                    triggered_clip_this_frame = "near_miss"

    # queue detection
    idle_people = [pid for pid in matched_ids if active_persons[pid].idle_start and (ts - active_persons[pid].idle_start) > 2.0]
    if len(idle_people) >= 2:
        cv2.putText(frame, f"QUEUE DETECTED: {len(idle_people)} people", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        queue_events.append({"ts": ts, "count": len(idle_people)})

    # save clip request
    if triggered_clip_this_frame:
        frames_list = list(video_buffer)
        save_clip_async(triggered_clip_this_frame, frames_list)

    # --- drawing ---
    if has_zones:
        for name, poly in polys.items():
            cv2.polylines(frame, [poly], True, (255,255,255), 1)

    # heatmap
    if show_heatmap and heatgrid.max() > 0:
        norm = (heatgrid / heatgrid.max() * 255).astype(np.uint8)
        hm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        hm = cv2.resize(hm, (w, h))
        frame = cv2.addWeighted(frame, 0.6, hm, 0.4, 0)

    # HUD
    frame_count += 1
    if frame_count % 10 == 0:
        fps = 10 / max(ts - fps_timer, 0.001)
        fps_timer = ts
        
    tot_idle = sum(p.total_idle for p in active_persons.values())
    tot_walk = sum(p.total_walk for p in active_persons.values()) / PX_PER_M
    cv2.putText(frame, f"FPS: {fps:.0f} | People: {len(active_persons)} | Cycles: {len(completed_cycles)} | Walk: {tot_walk:.1f}m | Violations: {len(violations)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("LineLens AI", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('h'): show_heatmap = not show_heatmap

cap.release()
cv2.destroyAllWindows()
if HAS_LEDS: led_buzzer.cleanup()

# --- save reports ---
print("\nSaving session data...")
os.makedirs("data", exist_ok=True)

# format the data exactly like the other scripts expect for the shift report
idle_data = [{"start_time": e["ts"]-e["dur"], "end_time": e["ts"]} for e in events_log if e["type"]=="idle"]
with open("data/idle.json", "w") as f: json.dump(idle_data, f)

tot_w = sum(p.total_walk for p in active_persons.values()) / PX_PER_M
with open("data/walk_positions.json", "w") as f: json.dump({"walk_m": round(tot_w, 2)}, f)

with open("data/ergo_violations.json", "w") as f: json.dump(violations, f)
with open("data/near_misses.json", "w") as f: json.dump(near_misses, f)
with open("data/queue_events.json", "w") as f: json.dump(queue_events, f)
with open("data/events.json", "w") as f: json.dump(events_log, f)
with open("data/cycles.json", "w") as f: json.dump(completed_cycles, f)

# calculate total average SOP drift
if completed_cycles:
    avg_drift = sum(c["drift_pct"] for c in completed_cycles) / len(completed_cycles)
else:
    avg_drift = 0.0
with open("data/sop_result.json", "w") as f: json.dump({"drift_pct": avg_drift}, f)

# save heatmap
if heatgrid.max() > 0:
    norm = (heatgrid / heatgrid.max() * 255).astype(np.uint8)
    hm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    hm = cv2.resize(hm, (640, 480))
    cv2.imwrite("data/heatmap.png", hm)
    print("Saved heatmap to data/heatmap.png")

# save spaghetti diagram
try:
    layout = cv2.imread("config/factory_layout1.jpg")
    if layout is not None:
        layout = cv2.resize(layout, (640, 480))
    else:
        layout = np.ones((480, 640, 3), dtype=np.uint8) * 30
except:
    layout = np.ones((480, 640, 3), dtype=np.uint8) * 30

for pt1, pt2 in all_walk_paths:
    cv2.line(layout, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

cv2.putText(layout, f"Total walk: {tot_w:.1f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.imwrite("data/spaghetti_diagram.png", layout)
print("Saved spaghetti diagram to data/spaghetti_diagram.png")

print("\nSaved all data to data/ folder!")

# auto-run the shift report
print("\nAuto-generating shift report (calling Groq for fixes)...")
_report_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics", "shift_report.py")
_result = subprocess.run([sys.executable, _report_script], capture_output=True, text=True)
if _result.stdout: print(_result.stdout)
if _result.returncode != 0:
    print("[shift_report error]:", _result.stderr[:500])
