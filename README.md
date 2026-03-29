# ParkMod: AI-Based Intelligent Parking Enforcement System

> **IEEE-BHTC Final-Year Engineering Project** | v1.0.0 | 2026

An AI-powered system that detects vehicles parked in no-parking zones using computer vision, monitors parking duration, detects violations, extracts number plates, and generates automated violation reports.

---

## 🎯 Features

- ✅ **Vehicle Detection** – YOLOv8 (car, motorcycle, bus, truck)
- ✅ **Multi-Object Tracking** – Centroid-based stable ID tracking
- ✅ **ROI No-Parking Zone** – Configurable polygon region
- ✅ **Bottom-Center Check** – Uses bottom-center of bbox for accurate ROI inclusion
- ✅ **Parking Timer** – Per-vehicle duration monitoring with entry/exit logic
- ✅ **Violation Detection** – Automatic flagging after configurable threshold
- ✅ **Evidence Capture** – Full frame + cropped vehicle saved on violation
- ✅ **License Plate OCR** – EasyOCR with regex cleaning
- ✅ **Report Generation** – CSV + JSON structured reports
- ✅ **API Simulation** – Simulated violation data transmission to authority
- ✅ **Web Dashboard** – Modern 5-tab Streamlit UI with metrics & charts
- ✅ **Demo Mode** – Works without GPU/model for presentation

---

## 🏗️ Architecture

```
Video Input
    │
    ▼
[YOLOv8 Detection] → filter vehicles only
    │
    ▼
[Centroid Tracking] → assign persistent vehicle IDs
    │
    ▼
[ROI Check] → is bottom-center inside no-parking zone?
    │
    ▼
[Timer Logic] → enter/exit timing, duration computation
    │
    ▼
[Violation Detection] → flag if duration > threshold
    │
    ├──→ [Evidence Capture] → save frame + crop
    ├──→ [OCR (EasyOCR)] → extract plate number
    ├──→ [Report Generator] → CSV + JSON
    └──→ [API Simulator] → send to authority (simulated)
    │
    ▼
[Streamlit Dashboard] → display everything
```

---

## 📁 Project Structure

```
IEEE-BHTC/
│
├── data/
│   ├── input_videos/       # Upload videos here
│   ├── output_frames/      # Annotated output frames
│   ├── evidence/           # Violation evidence images
│   └── reports/            # CSV + JSON reports
│
├── models/
│   └── yolov8n.pt          # YOLOv8 weights (download separately)
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Global settings & constants
│   ├── main.py             # Core pipeline + CLI entry point
│   ├── detector.py         # YOLOv8 vehicle detection
│   ├── tracker.py          # Centroid-based multi-object tracker
│   ├── roi_utils.py        # ROI polygon + point-in-polygon
│   ├── timer_logic.py      # Parking duration & violation rules
│   ├── ocr.py              # EasyOCR plate reader
│   ├── report_generator.py # CSV + JSON report generation
│   └── api_simulator.py    # Simulated enforcement API
│
├── app.py                  # Streamlit dashboard (5 tabs)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── run.sh                  # Quick launcher script
```

---

## 🚀 Installation

### 1. Clone / Download
```bash
cd IEEE-BHTC
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Download YOLOv8 Weights
```bash
# Auto-download:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Move to models/ folder:
mv yolov8n.pt models/
```
> **Without model weights, the system runs in demo mode automatically.**

---

## 📖 Usage

### Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### CLI Mode
```bash
# Demo mode (no video/model needed)
python -m src.main --demo --threshold 10

# Real video file
python -m src.main --video data/input_videos/parking.mp4 --threshold 10

# Save annotated output
python -m src.main --video parking.mp4 --save-video --threshold 15

# Headless (no display)
python -m src.main --video parking.mp4 --no-display --threshold 10
```

### Using run.sh (Linux/Mac)
```bash
chmod +x run.sh
./run.sh
```

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **📊 Dashboard** | Live metrics (total vehicles, violations, active tracking), violation timeline chart |
| **🎥 Detection** | Processed video frames with bounding boxes, ROI, vehicle IDs, duration, status |
| **🚨 Violations** | Filterable violation table with plate numbers, duration, status |
| **📄 Reports** | CSV + JSON report preview and download buttons |
| **🖼️ Evidence** | Grid view of captured violation evidence images |

---

## 📸 Screenshots

> *Screenshots will be added after deployment.*

| Dashboard | Detection | Violations |
|-----------|-----------|------------|
| ![Dashboard](#) | ![Detection](#) | ![Violations](#) |

---

## ⚙️ Configuration

Edit `src/config.py` to modify:

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_CONFIDENCE` | 0.40 | Detection confidence threshold |
| `DEFAULT_VIOLATION_THRESHOLD_SEC` | 10 | Seconds before violation |
| `TRACKER_MAX_DISTANCE` | 80 | Max centroid distance for matching |
| `TRACKER_MAX_DISAPPEARED` | 30 | Lost frames before track deletion |
| `OCR_GPU` | False | Enable GPU for EasyOCR |
| `DUPLICATE_COOLDOWN_SEC` | 30 | Prevent re-violation of same vehicle |

---

## 🔧 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| OpenCV | Video I/O & annotation |
| Ultralytics YOLOv8 | Vehicle detection |
| EasyOCR | License plate recognition |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Streamlit | Web dashboard |
| Shapely (optional) | Advanced polygon operations |

---

## 🔮 Future Scope

- 🔄 Real-time RTSP/IP camera streaming support
- 🧠 Deep SORT / ByteTrack for more robust tracking
- 📱 Mobile-responsive dashboard with alerts
- 🗄️ Database backend (SQLite/PostgreSQL)
- 📧 Email/SMS notification on violation
- 🌐 REST API for integration with city infrastructure
- 🚘 Vehicle make/model classification
- 📊 Analytics dashboard with heatmaps
- 🔐 Authentication & role-based access control

---

## 📝 License

This project is developed as part of the IEEE-BHTC final-year engineering program.

---

*Built with ❤️ using Python, YOLOv8, EasyOCR, and Streamlit*
