"""
ParkMod – src/roi_manager.py
Dynamic No-Parking Zone Manager.

Supports:
- Multiple named zones with per-zone thresholds and metadata
- Rectangle and polygon zone types
- JSON-based persistent storage (load/save)
- Active/inactive toggle per zone
- Zone CRUD (create, read, update, delete)
- OpenCV-based point-in-polygon checks
- Semitransparent overlay drawing with zone labels
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import ZONES_FILE, DEFAULT_VIOLATION_THRESHOLD_SEC, COLOR_ROI

Point = Tuple[int, int]
Polygon = List[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Return True if point lies inside polygon using OpenCV test."""
    poly_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, point, False) >= 0


def bbox_bottom_center(bbox: Tuple[int, int, int, int]) -> Point:
    """Return bbox bottom-center point used for ROI hit-testing."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


def is_vehicle_in_roi(
    bbox: Tuple[int, int, int, int], polygon: Polygon
) -> bool:
    """Compatibility helper for single-polygon ROI checks."""
    return point_in_polygon(bbox_bottom_center(bbox), polygon)


def draw_roi_on_frame(
    frame: np.ndarray,
    polygon: Polygon,
    label: str = "NO-PARKING ZONE",
    color: Tuple[int, int, int] = COLOR_ROI,
    alpha: float = 0.15,
) -> np.ndarray:
    """Draw a translucent polygon ROI overlay on a frame copy."""
    out = frame.copy()
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

    cx = int(np.mean([p[0] for p in polygon]))
    cy = int(min(p[1] for p in polygon)) - 10
    cv2.putText(
        out,
        label,
        (cx - 80, max(cy, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Zone Data Model
# ═══════════════════════════════════════════════════════════════

@dataclass
class Zone:
    """Represents a single no-parking zone."""

    id: str                                    # Unique zone ID
    name: str                                  # Display name
    zone_type: str                             # "rectangle" or "polygon"
    polygon: List[List[int]]                   # [[x,y], ...] ordered vertices
    threshold: float = DEFAULT_VIOLATION_THRESHOLD_SEC
    location: str = "Unspecified"
    department: str = "Traffic & Parking"
    active: bool = True                        # Can be toggled on/off
    color: List[int] = field(                  # BGR overlay color
        default_factory=lambda: [0, 255, 100]
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "zone_type": self.zone_type,
            "polygon": self.polygon,
            "threshold": self.threshold,
            "location": self.location,
            "department": self.department,
            "active": self.active,
            "color": self.color,
        }

    @staticmethod
    def from_dict(d: dict) -> "Zone":
        return Zone(
            id=d.get("id", str(uuid.uuid4())[:8]),
            name=d.get("name", "Unnamed"),
            zone_type=d.get("zone_type", "polygon"),
            polygon=d.get("polygon", []),
            threshold=d.get("threshold", DEFAULT_VIOLATION_THRESHOLD_SEC),
            location=d.get("location", "Unspecified"),
            department=d.get("department", "Traffic & Parking"),
            active=d.get("active", True),
            color=d.get("color", [0, 255, 100]),
        )

    @staticmethod
    def from_rectangle(
        id: str, name: str, x1: int, y1: int, x2: int, y2: int, **kwargs
    ) -> "Zone":
        """Create a zone from a top-left / bottom-right rectangle."""
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return Zone(id=id, name=name, zone_type="rectangle", polygon=poly, **kwargs)


# ═══════════════════════════════════════════════════════════════
# ROI Manager
# ═══════════════════════════════════════════════════════════════

class ROIManager:
    """
    Manages multiple dynamic No-Parking Zones.

    Zones are persisted in ``zones.json`` and loaded automatically
    on initialisation.  Only **active** zones participate in
    violation detection.
    """

    def __init__(self):
        self.zones: List[Zone] = []
        self.load_zones()

    # ── Persistence ───────────────────────────────────────────────

    def load_zones(self) -> None:
        """Load zones from the JSON file on disk."""
        self.zones.clear()
        if not ZONES_FILE.exists():
            return
        try:
            with open(ZONES_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for item in data.get("zones", []):
                self.zones.append(Zone.from_dict(item))
        except Exception as exc:
            print(f"[ROIManager] JSON load error: {exc}")

    def save_zones(self) -> None:
        """Write all zones back to the JSON file."""
        payload = {"zones": [z.to_dict() for z in self.zones]}
        with open(ZONES_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4)

    # ── CRUD ──────────────────────────────────────────────────────

    def add_zone(self, zone: Zone) -> None:
        """Add a new zone and persist."""
        self.zones.append(zone)
        self.save_zones()

    def update_zone(self, zone_id: str, updates: dict) -> bool:
        """
        Update fields of an existing zone.

        ``updates`` is a dict of field-name → new-value pairs.
        Returns True if the zone was found and updated.
        """
        for z in self.zones:
            if z.id == zone_id:
                for key, val in updates.items():
                    if hasattr(z, key):
                        setattr(z, key, val)
                self.save_zones()
                return True
        return False

    def delete_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID.  Returns True if found."""
        before = len(self.zones)
        self.zones = [z for z in self.zones if z.id != zone_id]
        if len(self.zones) < before:
            self.save_zones()
            return True
        return False

    def toggle_zone(self, zone_id: str) -> Optional[bool]:
        """Flip the active flag.  Returns new state, or None if not found."""
        for z in self.zones:
            if z.id == zone_id:
                z.active = not z.active
                self.save_zones()
                return z.active
        return None

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        for z in self.zones:
            if z.id == zone_id:
                return z
        return None

    @property
    def active_zones(self) -> List[Zone]:
        """Return only zones where ``active == True``."""
        return [z for z in self.zones if z.active]

    # ── Geometry checks ───────────────────────────────────────────

    def is_inside_any_zone(self, bbox: tuple) -> Optional[Zone]:
        """
        Check if the bottom-centre of *bbox* falls inside any
        **active** zone.

        Returns the first matching ``Zone`` or ``None``.
        """
        if not self.active_zones:
            return None

        x1, y1, x2, y2 = bbox
        bottom_centre = ((x1 + x2) / 2.0, float(y2))

        for z in self.active_zones:
            poly_np = np.array(z.polygon, dtype=np.int32)
            if cv2.pointPolygonTest(poly_np, bottom_centre, False) >= 0:
                return z
        return None

    # ── Drawing ───────────────────────────────────────────────────

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all **active** zones onto the frame as semi-transparent
        filled polygons with labels showing name + threshold.
        """
        if not self.active_zones:
            return frame

        overlay = frame.copy()
        for z in self.active_zones:
            color = tuple(z.color)
            poly_np = np.array(z.polygon, dtype=np.int32)

            # Filled polygon on overlay
            cv2.fillPoly(overlay, [poly_np], color)

            # Solid border on original frame
            cv2.polylines(frame, [poly_np], isClosed=True, color=color, thickness=2)

            # Label text at centre of polygon
            cx, cy = np.mean(poly_np, axis=0).astype(int)
            label = f"{z.name} ({z.threshold}s)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (cx - tw // 2 - 4, cy - th - 6),
                (cx + tw // 2 + 4, cy + 4),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, label,
                (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        return cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    def draw_zone_on_image(self, image: np.ndarray, zone: Zone) -> np.ndarray:
        """Draw a single zone on an image (used for preview)."""
        out = image.copy()
        overlay = out.copy()
        color = tuple(zone.color)
        poly_np = np.array(zone.polygon, dtype=np.int32)

        cv2.fillPoly(overlay, [poly_np], color)
        cv2.polylines(out, [poly_np], isClosed=True, color=color, thickness=3)

        cx, cy = np.mean(poly_np, axis=0).astype(int)
        cv2.putText(
            out, zone.name,
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
