"""
ParkMod – src/tracker_manager.py
Production tracker manager built on Ultralytics ByteTrack / BoT-SORT.

Responsibilities:
- Run detector+tracker and return stable tracked vehicle records.
- Smooth short-term occlusions by retaining recently seen track metadata.
- Enforce lost-track timeout cleanup for state handoff modules.

Returned track dict schema (per frame):
{
  "id": int,
  "class_id": int,
  "class_name": str,
  "confidence": float,
  "bbox": (x1, y1, x2, y2),
}
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    TRACKER_KEEP_LAST_BBOX,
    TRACKER_LOST_TIMEOUT_SEC,
    TRACKER_MIN_BOX_AREA,
    TRACKER_MIN_CONFIDENCE,
    TRACKER_PERSIST,
    TRACKER_TYPE,
    VEHICLE_CLASSES,
    YOLO_CONFIDENCE,
    YOLO_IOU_THRESH,
    YOLO_MODEL_PATH,
)

try:
    from ultralytics import YOLO

    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


@dataclass
class TrackRecord:
    """Last known metadata for a track ID."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    last_seen: float


class TrackerManager:
    """
    Vehicle tracker wrapper around Ultralytics native trackers.

    Supports:
    - ByteTrack ("bytetrack.yaml")
    - BoT-SORT ("botsort.yaml")
    """

    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self._model = None
        self._tracks: Dict[int, TrackRecord] = {}

        if not demo_mode:
            self._init_model()

    def _init_model(self) -> None:
        if not _HAS_ULTRALYTICS:
            raise RuntimeError(
                "ultralytics is required for TrackerManager. "
                "Install it or run with demo_mode=True."
            )
        self._model = YOLO(YOLO_MODEL_PATH)

    def track_frame(
        self,
        frame: np.ndarray,
        current_time: Optional[float] = None,
    ) -> List[dict]:
        """Run tracking on a frame and return active tracks."""
        if current_time is None:
            current_time = time.time()

        if self.demo_mode:
            return self._demo_tracks()

        results = self._model.track(
            frame,
            conf=max(YOLO_CONFIDENCE, TRACKER_MIN_CONFIDENCE),
            iou=YOLO_IOU_THRESH,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker=TRACKER_TYPE,
            persist=TRACKER_PERSIST,
            verbose=False,
        )

        tracks: List[dict] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Filter tiny boxes and low-confidence jitter.
                area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                if area < TRACKER_MIN_BOX_AREA:
                    continue
                if conf < TRACKER_MIN_CONFIDENCE:
                    continue

                rec = TrackRecord(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=VEHICLE_CLASSES.get(class_id, "vehicle"),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    last_seen=current_time,
                )
                self._tracks[track_id] = rec

                tracks.append(
                    {
                        "id": rec.track_id,
                        "class_id": rec.class_id,
                        "class_name": rec.class_name,
                        "confidence": rec.confidence,
                        "bbox": rec.bbox,
                    }
                )

        self._prune_lost_tracks(current_time)

        # Optional short occlusion smoothing: expose latest bbox for very recent misses.
        if TRACKER_KEEP_LAST_BBOX:
            visible_ids = {t["id"] for t in tracks}
            for tid, rec in self._tracks.items():
                age = current_time - rec.last_seen
                if tid in visible_ids:
                    continue
                if age <= TRACKER_LOST_TIMEOUT_SEC:
                    tracks.append(
                        {
                            "id": rec.track_id,
                            "class_id": rec.class_id,
                            "class_name": rec.class_name,
                            "confidence": rec.confidence,
                            "bbox": rec.bbox,
                        }
                    )

        return tracks

    def _prune_lost_tracks(self, current_time: float) -> None:
        stale_ids = [
            tid
            for tid, rec in self._tracks.items()
            if (current_time - rec.last_seen) > TRACKER_LOST_TIMEOUT_SEC
        ]
        for tid in stale_ids:
            del self._tracks[tid]

    def get_track(self, track_id: int) -> Optional[TrackRecord]:
        return self._tracks.get(track_id)

    def active_ids(self, current_time: Optional[float] = None) -> List[int]:
        if current_time is None:
            current_time = time.time()
        return [
            tid
            for tid, rec in self._tracks.items()
            if (current_time - rec.last_seen) <= TRACKER_LOST_TIMEOUT_SEC
        ]

    def reset(self) -> None:
        self._tracks.clear()

    @staticmethod
    def _demo_tracks() -> List[dict]:
        return [
            {
                "id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.91,
                "bbox": (100, 120, 320, 290),
            }
        ]
