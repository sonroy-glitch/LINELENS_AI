# LineLens AI

**Privacy-first computer-vision analytics for the factory floor.**

LineLens AI watches a video feed of a work area and, in real time, tracks multiple people at once to measure how work actually happens — idle time, walking distance, ergonomic risk, zone/process-order compliance, work cycles, near-misses, and bottlenecks. At the end of a shift it produces an interactive report with charts, a heatmap, a walk-path "spaghetti" diagram, and an AI-generated list of the top process improvements. A live web dashboard mirrors the latest results.

Faces are blurred before anything else runs, so no identifiable footage is ever stored or uploaded.

---

## What it does

| Capability | Description |
|---|---|
| **Multi-person tracking** | Detects and follows several workers simultaneously with a custom centroid tracker (EMA-smoothed, nearest-neighbour matching). |
| **Idle detection** | Flags workers who stop moving, using an anchor-position method that resists slow-walk "cheating" and camera jitter. |
| **Walk distance** | Measures distance travelled per person in metres via pixel-per-metre calibration. |
| **Ergonomics** | Computes trunk-bend, deep-squat, and sustained-overhead-reach risks from body-joint angles. |
| **Zones & SOP compliance** | Assigns each worker to a floor zone and scores how far their zone sequence drifts from the expected standard operating procedure. |
| **Cycle segmentation** | Detects completed work cycles and their duration. |
| **Near-miss detection** | Warns when two people come dangerously close. |
| **Queue / bottleneck detection** | Flags when multiple workers are idle at once. |
| **Alerts** | Fires LED / buzzer signals (Raspberry Pi GPIO) and auto-saves a 10-second video clip around each event. |
| **Heatmap & spaghetti diagram** | Visualises where people spend time and how they move. |
| **Shift report** | Self-contained HTML report with charts + top-3 AI process fixes. |
| **Live dashboard** | React web app that pulls the latest report from the cloud and refreshes automatically. |

---

## Architecture

```
video feed
   │
   ▼
face blur (privacy)  ──►  YOLOv8-pose (person + 17 keypoints)
                              │
                              ▼
                 custom multi-person tracker
                              │
        ┌──────────┬──────────┼──────────┬───────────┐
        ▼          ▼          ▼          ▼           ▼
     idle /     ergonomics   zones /    near-miss   cycles
     walking                  SOP        / queue
        └──────────┴──────────┴──────────┴───────────┘
                              │
                              ▼
                   session data (JSON)
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                 ▼
      shift report      Supabase (cloud)   heatmap /
      (HTML + Groq                          spaghetti
       AI fixes)                            diagram
                              │
                              ▼
                     React dashboard
```

---

## Project layout

```
run_all.py              main real-time pipeline (detection + all analytics)
detection/              per-metric logic: idle, ergonomics, zones, SOP, cycles, near-miss, queue, walking
tracking/               heatmap, spaghetti diagram, trajectory tracking
vision/                 pose estimation, face blurring, skeleton rendering
calibration/            zone drawing + perspective calibration tools
alerts/                 LED/buzzer, event detection, auto clip saver
analytics/              shift report generator + graphs
upload_to_supabase.py   pushes session results to the cloud database
dashboard/              React + Vite live dashboard
config/                 zone definitions
data/                   session output (JSON, images)
```

---

## Setup

### 1. Python pipeline

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root (this file is **git-ignored** and never committed):

```
GROQ_API_KEY=your_groq_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

Calibrate the zones for your camera view (one-time):

```bash
python calibration/zone_drawing.py
```

Run the full pipeline on a video source:

```bash
python run_all.py
```

Press **q** to quit and generate the shift report, **h** to toggle the live heatmap.

### 2. Dashboard

```bash
cd dashboard
npm install
npm run dev
```

---

## What is my own work vs. third-party

LineLens AI is built on well-known open-source tools, but the analytics engine — everything that turns raw body-keypoints into safety and productivity insight — is my own code.

**Third-party building blocks (used under their licences):**
- **YOLOv8-pose** (Ultralytics) — pre-trained model that detects people and their 17 body keypoints. I use it as-is; I did **not** train it.
- **MediaPipe** (Google) — face detection, used only to blur faces.
- **OpenCV**, **NumPy**, **Shapely**, **Matplotlib** — video I/O, math, geometry, plotting.
- **Groq API** (Llama 3.3 70B) — generates the shift's top-3 written recommendations.
- **Supabase** — cloud database for the dashboard.
- **React**, **Vite**, **Chart.js** — dashboard front-end.

**My original work:**
- The entire real-time pipeline that fuses multi-person detection with every analytic ([run_all.py](run_all.py)).
- The custom multi-person **tracker** (EMA smoothing + nearest-neighbour matching) and per-person state machine.
- All **metric logic**, derived from keypoint geometry: idle/anti-drift detection, walk-distance calibration, ergonomic bend/squat/overhead angles, zone assignment, SOP-drift scoring, cycle segmentation, near-miss proximity, and queue detection.
- The **heatmap** and **spaghetti-diagram** generators.
- The event-driven **alerts** (GPIO) and rolling-buffer **clip saver**.
- The **shift-report** generator, the Groq prompt design, and the rule-based fallback.
- The **Supabase** upload pipeline and the **React dashboard** UI.

> A full breakdown is in [LineLens_AI_Originality.docx](LineLens_AI_Originality.docx).

---

## Built for the Congressional App Challenge 2026

LineLens AI addresses workplace safety and productivity — helping small manufacturers spot ergonomic risks and inefficiencies without buying expensive proprietary systems, and without sacrificing worker privacy.
