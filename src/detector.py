import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    pass

from src.config import YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU_THRESH, VEHICLE_CLASSES, TRACKER_TYPE

class VehicleTracker:
    """Wrapper for native Ultralytics ByteTrack/BoT-SORT Tracker logic."""
    
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        self.model = YOLO(YOLO_MODEL_PATH) if not demo_mode else None
        
    def track_frame(self, frame: np.ndarray) -> list:
        if self.demo_mode:
            return [{"id": 1, "bbox": (10, 10, 100, 100), "class": "car", "conf": 0.9}]
            
        results = self.model.track(
            frame,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU_THRESH,
            tracker=TRACKER_TYPE,
            classes=list(VEHICLE_CLASSES.keys()),
            persist=True,
            verbose=False
        )
        
        active_tracks = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue  # Wait for stable track ID
                    
                tid = int(box.id[0])
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                active_tracks.append({
                    "id": tid,
                    "class_id": cls_id,
                    "class_name": VEHICLE_CLASSES.get(cls_id, "vehicle"),
                    "confidence": float(box.conf[0]),
                    "bbox": (x1, y1, x2, y2)
                })
        return active_tracks
