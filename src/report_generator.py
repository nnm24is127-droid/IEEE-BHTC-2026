"""
ParkMod – src/report_generator.py
Generates structured violation reports in CSV and JSON formats.
Stores evidence images in the evidence/ folder.
"""

from __future__ import annotations

import csv
import json
import io
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

from src.config import (
    CSV_REPORT_PATH,
    JSON_REPORT_PATH,
    EVIDENCE_DIR,
    PROJECT_NAME,
    PROJECT_VERSION,
)


# CSV field order
REPORT_FIELDS = [
    "vehicle_id",
    "plate_number",
    "timestamp",
    "duration_sec",
    "location",
    "zone_id",
    "zone_name",
    "image_path",
    "evidence_saved",
    "status",
]


class ReportGenerator:
    """
    Generates and manages violation reports.

    Features:
    - Appends violation records in real-time
    - Saves CSV and JSON reports
    - Captures and stores evidence images
    - Provides summary statistics

    Parameters
    ----------
    location : str
        Human-readable location label.
    csv_path : Path
        Output CSV file path.
    json_path : Path
        Output JSON file path.
    evidence_dir : Path
        Directory for evidence screenshots.
    """

    def __init__(
        self,
        location: str = "Main Parking Lot",
        csv_path: Path = CSV_REPORT_PATH,
        json_path: Path = JSON_REPORT_PATH,
        evidence_dir: Path = EVIDENCE_DIR,
    ):
        self.location = location
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path)
        self.evidence_dir = Path(evidence_dir)
        self._records: List[dict] = []

        # Ensure directories exist
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    # ── Evidence Capture ───────────────────────────────────────────────────

    def save_evidence(
        self,
        frame: np.ndarray,
        vehicle_id: int,
        bbox: tuple,
    ) -> str:
        """
        Save both the full frame and cropped vehicle image as evidence.
        Returns the path to the saved evidence image.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname_full = f"violation_VH{vehicle_id:04d}_{timestamp}_full.jpg"
        fname_crop = f"violation_VH{vehicle_id:04d}_{timestamp}_crop.jpg"

        full_path = self.evidence_dir / fname_full
        crop_path = self.evidence_dir / fname_crop

        # Save full frame
        cv2.imwrite(str(full_path), frame)

        # Save cropped vehicle
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
        x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)

        return str(full_path)

    # ── Record Management ──────────────────────────────────────────────────

    def add_violation(
        self,
        vehicle_id: int,
        plate_number: str,
        duration_sec: float,
        image_path: str = "",
        zone_id: str = "",
        zone_name: str = "",
        evidence_saved: bool = False,
    ) -> dict:
        """Add a new violation record."""
        record = {
            "vehicle_id": vehicle_id,
            "plate_number": plate_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration_sec, 2),
            "location": self.location,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "image_path": image_path,
            "evidence_saved": bool(evidence_saved),
            "status": "Violation",
        }
        self._records.append(record)
        self._write_csv()
        self._write_json()
        return record

    # ── CSV ────────────────────────────────────────────────────────────────

    def _write_csv(self) -> None:
        """Write all records to CSV."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(self._records)

    def to_csv_bytes(self) -> bytes:
        """Return the current report as CSV bytes for download."""
        buf = io.StringIO()
        buf.write(f"# {PROJECT_NAME} v{PROJECT_VERSION}\n")
        buf.write(
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        buf.write(f"# Location: {self.location}\n")
        buf.write("#\n")
        df = self.to_dataframe()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    # ── JSON ───────────────────────────────────────────────────────────────

    def _write_json(self) -> None:
        """Write all records to JSON."""
        payload = {
            "project": PROJECT_NAME,
            "version": PROJECT_VERSION,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": self.location,
            "total": len(self._records),
            "violations": self._records,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ── DataFrame ──────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a pandas DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=REPORT_FIELDS)
        return pd.DataFrame(self._records)

    def load_from_csv(self) -> pd.DataFrame:
        """Load existing CSV report from disk."""
        if not self.csv_path.exists():
            return pd.DataFrame(columns=REPORT_FIELDS)
        try:
            return pd.read_csv(self.csv_path, comment="#")
        except Exception:
            return pd.DataFrame(columns=REPORT_FIELDS)

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Compute summary statistics."""
        df = self.to_dataframe()
        if df.empty:
            return {"total": 0, "avg_duration": 0, "max_duration": 0}

        return {
            "total": len(df),
            "avg_duration": round(df["duration_sec"].mean(), 1),
            "max_duration": round(df["duration_sec"].max(), 1),
            "unique_plates": df["plate_number"].nunique(),
        }

    def text_summary(self) -> str:
        """Plain-text summary report."""
        s = self.summary()
        if s["total"] == 0:
            return "No violations recorded."
        return "\n".join(
            [
                "=" * 50,
                f"  {PROJECT_NAME}",
                "=" * 50,
                f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"  Location  : {self.location}",
                "-" * 50,
                f"  Total Violations  : {s['total']}",
                f"  Unique Plates     : {s.get('unique_plates', 0)}",
                f"  Avg Duration      : {s['avg_duration']}s",
                f"  Max Duration      : {s['max_duration']}s",
                "=" * 50,
            ]
        )

    # ── Utilities ──────────────────────────────────────────────────────────

    def get_records(self) -> List[dict]:
        return list(self._records)

    def get_evidence_images(self) -> List[Path]:
        """List all evidence images in the evidence directory."""
        if not self.evidence_dir.exists():
            return []
        return sorted(self.evidence_dir.glob("violation_*.jpg"))

    def reset(self) -> None:
        """Clear all records."""
        self._records.clear()
        if self.csv_path.exists():
            self._write_csv()
        if self.json_path.exists():
            self._write_json()
