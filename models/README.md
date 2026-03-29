# Place YOLOv8 Weights Here

Download the YOLOv8 nano model weights from Ultralytics:

```bash
# Option 1 – automatic download (requires ultralytics installed)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Then move/copy yolov8n.pt to this models/ folder

# Option 2 – direct download
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Expected file: `models/yolov8n.pt`

> **Note:** If this file is missing, ParkMod automatically switches to **demo mode**
> (synthetic detections) so the dashboard and CLI still work without any GPU or model.

## Available YOLOv8 Variants

| Model        | Size  | mAP  | Speed (CPU) |
|-------------|-------|------|-------------|
| yolov8n.pt  | 6 MB  | 37.3 | Fast        |
| yolov8s.pt  | 22 MB | 44.9 | Moderate    |
| yolov8m.pt  | 50 MB | 50.2 | Slow        |

For a parking-lot deployment, `yolov8n.pt` is recommended for real-time performance.
