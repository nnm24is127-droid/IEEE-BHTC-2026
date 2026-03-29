"""
ParkMod – src/main.py
Core detection pipeline that wires all modules together.

Architecture:
  Video/Webcam → YOLOv8 + ByteTrack → Dynamic ROI Check
  → Violation State Machine → 2-Stage ALPR → Report + Email + API
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Ensure project root is on sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import WEBCAM_INDEX, PROJECT_NAME
from src.tracker_manager import TrackerManager
from src.roi_manager import ROIManager
from src.violation_engine import ViolationEngine
from src.alpr import ALPR
from src.report_generator import ReportGenerator
from src.email_notifier import send_violation_email


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

class ParkModPipeline:
    """End-to-end parking enforcement pipeline."""

    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.tracker = TrackerManager(demo_mode)
        self.roi = ROIManager()
        self.engine = ViolationEngine()
        self.alpr = ALPR(demo_mode)
        self.report = ReportGenerator("Unified Report Manager")
        self.frame_count = 0
        self.total_vehicles = 0

    # ── reload zones at runtime (useful when UI saves new zones) ──
    def reload_zones(self):
        self.roi.load_zones()

    # ── Single frame ──────────────────────────────────────────────

    def process_frame(
        self, frame: np.ndarray, current_time: float
    ) -> np.ndarray:
        # 1. Detect + Track
        tracks = self.tracker.track_frame(frame, current_time=current_time)
        if tracks:
            self.total_vehicles = max(
                self.total_vehicles,
                max(t["id"] for t in tracks),
            )

        # 2. Evaluate violations against **active** zones only
        new_violations = self.engine.process_frame(
            tracks, self.roi.active_zones, current_time
        )

        # 3. ALPR: keep reading plates for in-zone vehicles
        for veh in tracks:
            vid = veh["id"]
            bbox = veh["bbox"]
            state = self.engine.get_state(vid)
            if state and state.active_zone_id:
                best = self.alpr.read_plate(frame, bbox, vid)
                self.engine.set_best_plate(vid, best)

        # 4. Handle new violations
        for state in new_violations:
            if state.reported:
                continue

            plate = self.alpr.get_best_plate(state.id)
            self.engine.set_best_plate(state.id, plate)

            # Find bbox for this vehicle
            bbox = None
            for t in tracks:
                if t["id"] == state.id:
                    bbox = t["bbox"]
                    break

            img_path = ""
            if bbox:
                img_path = self.report.save_evidence(frame, state.id, bbox)

            self.engine.mark_reported(
                vehicle_id=state.id,
                evidence_path=img_path if img_path else None,
                best_plate=plate,
            )

            self.report.add_violation(
                vehicle_id=state.id,
                plate_number=plate,
                duration_sec=state.accumulated_time,
                image_path=img_path,
                zone_id=state.zone_id,
                zone_name=state.zone_name,
                evidence_saved=bool(img_path),
            )

            send_violation_email(
                vehicle_id=state.id,
                plate_number=plate,
                duration_sec=state.accumulated_time,
                location=state.zone_name or "Unified Report Manager",
                evidence_path=img_path if img_path else None,
            )
            print(
                f"[VIOLATION] Veh:{state.id}  "
                f"Duration:{state.accumulated_time:.1f}s  "
                f"Zone:{state.zone_name}  "
                f"Plate:{plate}"
            )

        # 5. Annotate
        self.frame_count += 1
        return self._annotate(frame, tracks)

    # ── Drawing ───────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        out = self.roi.draw_zones(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for v in tracks:
            vid = v["id"]
            x1, y1, x2, y2 = v["bbox"]
            state = self.engine.get_state(vid)

            color = (150, 150, 150)
            status_text = "Outside"
            plate = "UNKNOWN"
            dur = 0.0

            if state:
                plate = state.best_current_plate
                dur = state.accumulated_time
                if state.status == "Violation":
                    color = (0, 0, 255)
                    status_text = "!! VIOLATION !!"
                elif state.status == "Warning":
                    color = (0, 165, 255)
                    status_text = "Warning"
                elif state.status == "Compliant":
                    color = (0, 200, 0)
                    status_text = "Compliant"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{vid} | {v['class_name']}"
            cv2.putText(
                out, label, (x1, y1 - 5),
                font, 0.5, color, 1, cv2.LINE_AA,
            )

            info = f"Plate:{plate} | {dur:.1f}s"
            cv2.putText(
                out, info, (x1, y2 + 15),
                font, 0.45, color, 1, cv2.LINE_AA,
            )

            if status_text not in ("Outside",):
                cv2.putText(
                    out, status_text, (x1, y1 - 22),
                    font, 0.6, color, 2, cv2.LINE_AA,
                )

        return out

    # ── CLI video loop ────────────────────────────────────────────

    def run_video(self, source=None, no_display: bool = False):
        cap = cv2.VideoCapture(source if source else WEBCAM_INDEX)

        use_video_time = bool(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if use_video_time:
                pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if pos_sec <= 0:
                    pos_sec = frame_idx / fps
                current_time = pos_sec
            else:
                current_time = time.time()

            annotated = self.process_frame(frame, current_time)
            frame_idx += 1
            if not no_display:
                cv2.imshow("ParkMod – press Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ParkMod CLI – run detection on video or webcam"
    )
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    pipe = ParkModPipeline(demo_mode=args.demo)
    pipe.run_video(source=args.video, no_display=args.no_display)
