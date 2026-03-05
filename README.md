# ShelfEye

**Software-only retail attention intelligence.** Stores use their existing IP cameras and computers; ShelfEye tells them where shoppers look, how long they look, and how many unique people visit each shelf zone. Monthly subscription, live web dashboard with heatmaps and AI recommendations — no new hardware.

---

## Overview

ShelfEye runs as a Python desktop app on the store’s existing machine. OpenCV reads the camera at 30 FPS, MediaPipe detects faces and **468 facial landmarks** per face, and our engine computes head direction:

- **Up/down (pitch):** angle from forehead to chin  
- **Left/right (yaw):** nose position relative to ear-to-ear  

Everything is **normalized by face size**, so tracking works whether the camera is 1 m or 5 m away. An **auto-calibration** over the first 30 frames learns each person’s neutral head pose so accuracy holds for anyone in front of the camera.

Head direction is mapped to **nine shelf zones** — three aisles × three heights (top, eye level, bottom). A **person tracker** assigns a unique ID to each face and follows them across the frame. Time in a zone is classified as:

| Category | Duration | Meaning |
|----------|----------|---------|
| *Pass* | &lt; 3 s | Ignored (just passing through) |
| **Glance** | 3–8 s | Quick look |
| **Browse** | 8–30 s | Meaningful look |
| **Dwell** | 30+ s | Serious consideration — highest value signal |

That gives stores not just visit counts but **depth of engagement** (e.g. dwells vs glances).

Data is written to a JSON file every 5 seconds. A **Flask dashboard** reads it every 3 seconds and shows:

- **Top:** four stat cards — total people, active now, total visits, most popular aisle  
- **Middle:** 3×3 attention heatmap (red = high, orange = medium, green = low) and stacked bar chart (glance / browse / dwell per aisle)  
- **Bottom:** three aisle cards with full breakdown (unique people, glance, browse, dwell)

---

## Tech stack (current)

- **Python 3.11**, **OpenCV**, **MediaPipe 0.10.9**, **Flask**
- **Windows** — `cv2.CAP_DSHOW` for camera compatibility; MediaPipe is pinned to **0.10.9** (newer versions have DLL issues on Windows)
- **Core tracking engine:** fully working and tested on a laptop webcam

---

## Roadmap

**Next**

1. **AWS cloud** — persist data online so store owners can view the dashboard from any device, anywhere  
2. **Login & multi-tenant** — one account per store  
3. **Real IP camera support** — RTSP + ONVIF auto-discovery  
4. **Windows installer** — stores set up in under 5 minutes with no technical knowledge  

**Later**

- **ML features:** anomaly detection (unusual attention patterns), emotion detection (e.g. AWS Rekognition), layout recommendations (predict impact of moving products), traffic prediction (busy/quiet periods ~24 h ahead)  
- **Brand partner portal** — FMCG brands (e.g. Coca-Cola, Unilever) pay for attention data on their products across stores → second revenue stream alongside store subscriptions  

---


---

## Prerequisites

- **Python 3.11** (3.8+ may work)
- **Webcam** (or compatible camera)
- **Windows** (tested; macOS/Linux may need camera/index changes)

---

## Setup

1. **Clone or download** this repo.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or manually:
   ```bash
   pip install flask opencv-python "mediapipe==0.10.9"
   ```

---

## How to run

Run **two processes**: the tracker (camera + logging) and the dashboard.

### 1. Start the tracker

```bash
python test.py
```

- **First ~3 seconds:** Look straight at the camera for calibration.  
- **Then:** Move your head; the overlay shows the current zone (e.g. “Aisle 2 - Eye Level”).  
- **Keys:** **Q** quit · **S** save · **R** reset session · **C** recalibrate  

Stats auto-save every 5 seconds when there’s activity.

### 2. Start the dashboard

In a **second terminal**:

```bash
python dashboard.py
```

Open **http://localhost:5000**. The page polls `/data` every 3 seconds.

---

## Project structure

```
shelfeye/
├── dashboard.py       # Flask app: serves UI and /data from attention_log.json
├── test.py            # Camera + MediaPipe + tracking + zone logic + logging
├── attention_log.json # Generated: session stats (written by test.py)
├── templates/
│   └── index.html     # Dashboard UI (stats, heatmap, charts)
├── requirements.txt
└── README.md
```

---

## Configuration (in code)

In `test.py`:

- **Sensitivity:** `YAW_THRESHOLD`, `PITCH_THRESHOLD` (smaller = more sensitive)  
- **Visit thresholds (seconds):** `PASS_THRESHOLD` (3), `GLANCE_THRESHOLD` (8), `BROWSE_THRESHOLD` (30)  
- **Stability:** `STABILITY_FRAMES` (frames in zone before counting)  
- **Calibration:** `CALIBRATION_NEEDED` (default 30 frames)  

Dashboard port in `dashboard.py` (default `5000`).

---

## Publish to GitHub

**If you don’t have Git:** Install it from [git-scm.com](https://git-scm.com/download/win), then restart your terminal.

1. **Initialize and commit** (in PowerShell or Command Prompt):
   ```bash
   cd c:\Users\patel\Desktop\shelfeye
   git init
   git add .
   git commit -m "Initial commit: ShelfEye retail attention intelligence"
   ```

2. **Create a new repository** on [GitHub](https://github.com/new):
   - Name it `shelfeye` (or any name you like).
   - Leave “Add a README” unchecked — this project already has one.
   - Click **Create repository**.

3. **Connect and push** (replace `YOUR_USERNAME` with your GitHub username):
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/shelfeye.git
   git branch -M main
   git push -u origin main
   ```
   If you use SSH:
   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/shelfeye.git
   git branch -M main
   git push -u origin main
   ```

4. **Later updates:**
   ```bash
   git add .
   git commit -m "Your message"
   git push
   ```

---

## License

Use and modify as you like. No warranty.
