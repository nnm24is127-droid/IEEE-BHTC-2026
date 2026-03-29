from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_VIDEO_DIR = DATA_DIR / "input_videos"
OUTPUT_FRAME_DIR = DATA_DIR / "output_frames"
EVIDENCE_DIR = DATA_DIR / "evidence"
PLATE_CROPS_DIR = DATA_DIR / "plate_crops"
REPORTS_DIR = DATA_DIR / "reports"
ZONES_DIR = DATA_DIR / "zones"
MODELS_DIR = BASE_DIR / "models"

for d in [INPUT_VIDEO_DIR, OUTPUT_FRAME_DIR, EVIDENCE_DIR, PLATE_CROPS_DIR, REPORTS_DIR, ZONES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Files
ZONES_FILE = ZONES_DIR / "zones.json"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
PLATE_MODEL_PATH = MODELS_DIR / "plate_detector.pt" # Option for ALPR
CSV_REPORT_PATH = REPORTS_DIR / "violations_report.csv"
JSON_REPORT_PATH = REPORTS_DIR / "violations_report.json"

# Detector Settings
YOLO_CONFIDENCE = 0.40
YOLO_IOU_THRESH = 0.45
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Tracking
TRACKER_TYPE = "bytetrack.yaml" # Available native trackers: bytetrack.yaml, botsort.yaml
TRACKER_PERSIST = True
TRACKER_MIN_CONFIDENCE = 0.30
TRACKER_MIN_BOX_AREA = 900
TRACKER_LOST_TIMEOUT_SEC = 1.5
TRACKER_KEEP_LAST_BBOX = True

# Violation Settings
DEFAULT_VIOLATION_THRESHOLD_SEC = 10
LOST_TRACK_TIMEOUT_SEC = 5
DUPLICATE_COOLDOWN_SEC = 60

# OCR / ALPR
OCR_LANGUAGES = ["en"]
OCR_GPU = False
OCR_MIN_CONF = 0.15
OCR_FRAME_GAP = 6

# Camera / Video
WEBCAM_INDEX = 0
TARGET_FPS = 30

# App Info
PROJECT_NAME = "ParkMod: AI-Based Intelligent Parking Enforcement System"
PROJECT_VERSION = "2.0.0"

# Colors
COLOR_VIOLATION = (0, 0, 255)
COLOR_WARNING = (0, 165, 255)
COLOR_COMPLIANT = (0, 200, 0)
COLOR_ROI = (0, 255, 100)
COLOR_TEXT = (255, 255, 255)
FONT_SCALE = 0.55
BOX_THICKNESS = 2
