"""
ParkMod – src/violation_engine.py
Parking violation state-machine.

Each tracked vehicle is modelled as a ``VehicleState`` that
accumulates time while it stays inside an active zone.

Features:
- Per-zone configurable violation threshold
- Grace period for lost-track (LOST_TRACK_TIMEOUT_SEC)
- Explicit per-track finite state machine
- Duplicate violation prevention (once reported, stays reported)
- Zone-aware: stores which zone triggered the violation
- Evidence flags and best-plate attachment for reporting
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.roi_manager import Zone
from src.config import LOST_TRACK_TIMEOUT_SEC


# ── State constants ─────────────────────────────────────────────

STATE_OUTSIDE_ROI = "outside_roi"
STATE_ENTERED_ROI = "entered_roi"
STATE_ACTIVE_TIMER = "active_timer"
STATE_VIOLATION_TRIGGERED = "violation_triggered"
STATE_EXITED_ROI = "exited_roi"


# ═══════════════════════════════════════════════════════════════
# Vehicle State Data Model
# ═══════════════════════════════════════════════════════════════

@dataclass
class VehicleState:
    """Mutable state record for a single tracked vehicle."""

    id: int
    first_seen: float                           # Absolute time first observed
    last_seen: float                            # Absolute time last frame seen
    entry_time: float                           # Clock at last delta computation

    # Zone association
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    zone_threshold: float = 10.0
    active_zone_id: Optional[str] = None          # Backward compatibility
    active_zone_name: Optional[str] = None        # Backward compatibility
    active_zone_threshold: float = 10.0           # Backward compatibility

    # Accumulated parking duration (seconds inside ROI)
    accumulated_time: float = 0.0

    # Violation flags
    is_violation: bool = False
    reported: bool = False

    # Evidence
    evidence_saved: bool = False
    evidence_path: Optional[str] = None
    best_plate: str = "UNKNOWN"
    best_current_plate: str = "UNKNOWN"  # Backward compatibility

    # FSM
    state: str = STATE_OUTSIDE_ROI
    entered_roi_at: Optional[float] = None
    exited_roi_at: Optional[float] = None
    violation_triggered_at: Optional[float] = None

    # Display status string
    status: str = "Outside"


# ═══════════════════════════════════════════════════════════════
# Violation Engine
# ═══════════════════════════════════════════════════════════════

class ViolationEngine:
    """
    State-machine tracker for robust illegal-parking evaluation.

    Each vehicle track ID maps to a ``VehicleState``.  The engine
    accumulates time when the vehicle is inside an active zone and
    triggers a violation once accumulated_time >= zone.threshold.

    Lost tracks are cleaned up after ``LOST_TRACK_TIMEOUT_SEC``.
    """

    def __init__(self):
        self.states: Dict[int, VehicleState] = {}

    # ── Per-frame evaluation ──────────────────────────────────────

    def process_frame(
        self,
        tracked_vehicles: List[dict],
        active_zones: List[Zone],
        current_time: Optional[float] = None,
    ) -> List[VehicleState]:
        """
        Evaluate all tracked vehicles against the active zones.

        Parameters
        ----------
        tracked_vehicles : list of dict
            Each dict has keys: id, bbox, class_name, confidence
        active_zones : list of Zone
            Only **active** zones from ROIManager.
        current_time : float or None
            Explicit clock (for video-mode); defaults to wall-clock.

        Returns
        -------
        new_violations : list of VehicleState that JUST violated
        """
        if current_time is None:
            current_time = time.time()

        # ── Clean up long-lost tracks ─────────────────────────────
        for vid in list(self.states):
            state = self.states[vid]
            age = current_time - state.last_seen

            # Keep unreported violations for downstream report save;
            # prune non-violations or already reported violations.
            if age > LOST_TRACK_TIMEOUT_SEC:
                if (not state.is_violation) or state.reported:
                    del self.states[vid]

        # ── Evaluate each visible vehicle ─────────────────────────
        new_violations: List[VehicleState] = []

        for veh in tracked_vehicles:
            vid = veh["id"]
            bbox = veh["bbox"]

            # Which zone (if any) is this vehicle inside?
            zone_hit = _zone_for_bbox(bbox, active_zones)

            # --- First appearance → create state ---
            if vid not in self.states:
                self.states[vid] = VehicleState(
                    id=vid,
                    first_seen=current_time,
                    last_seen=current_time,
                    entry_time=current_time,
                    zone_id=zone_hit.id if zone_hit else None,
                    zone_name=zone_hit.name if zone_hit else None,
                    zone_threshold=(
                        zone_hit.threshold if zone_hit else 10.0
                    ),
                )

            state = self.states[vid]
            delta = current_time - state.entry_time
            state.entry_time = current_time
            state.last_seen = current_time

            if zone_hit is not None:
                # ── Vehicle is inside a zone ──────────────────────
                state.zone_id = zone_hit.id
                state.zone_name = zone_hit.name
                state.zone_threshold = zone_hit.threshold
                state.active_zone_id = zone_hit.id
                state.active_zone_name = zone_hit.name
                state.active_zone_threshold = zone_hit.threshold
                state.accumulated_time += delta

                threshold = zone_hit.threshold

                if state.entered_roi_at is None:
                    state.entered_roi_at = current_time
                    state.state = STATE_ENTERED_ROI
                    state.status = "Compliant"
                elif not state.is_violation:
                    state.state = STATE_ACTIVE_TIMER

                if (
                    state.accumulated_time >= threshold
                    and not state.is_violation
                    and not state.reported
                ):
                    state.is_violation = True
                    state.state = STATE_VIOLATION_TRIGGERED
                    state.violation_triggered_at = current_time
                    state.status = "Violation"
                    new_violations.append(state)
                elif not state.is_violation:
                    if state.accumulated_time >= threshold * 0.5:
                        state.status = "Warning"
                    else:
                        state.status = "Compliant"
            else:
                # ── Vehicle outside all zones ─────────────────────
                if state.entered_roi_at is not None:
                    state.state = STATE_EXITED_ROI
                    state.exited_roi_at = current_time
                else:
                    state.state = STATE_OUTSIDE_ROI

                state.accumulated_time = 0.0
                state.zone_id = None
                state.zone_name = None
                state.zone_threshold = 10.0
                state.active_zone_id = None
                state.active_zone_name = None
                state.active_zone_threshold = 10.0
                state.entered_roi_at = None
                state.status = "Outside"

        return new_violations

    # ── Utilities ─────────────────────────────────────────────────

    def get_state(self, vehicle_id: int) -> Optional[VehicleState]:
        return self.states.get(vehicle_id)

    def mark_reported(
        self,
        vehicle_id: int,
        evidence_path: Optional[str] = None,
        best_plate: Optional[str] = None,
    ) -> bool:
        """Mark a violation state as reported and persist evidence/plate metadata."""
        state = self.states.get(vehicle_id)
        if not state:
            return False

        state.reported = True
        if evidence_path:
            state.evidence_path = evidence_path
            state.evidence_saved = True
        if best_plate:
            state.best_plate = best_plate
            state.best_current_plate = best_plate
        return True

    def set_best_plate(self, vehicle_id: int, plate: str) -> None:
        state = self.states.get(vehicle_id)
        if state:
            state.best_plate = plate
            state.best_current_plate = plate

    def reset(self) -> None:
        self.states.clear()


# ═══════════════════════════════════════════════════════════════
# Geometry helper (module-level, avoids repeated imports)
# ═══════════════════════════════════════════════════════════════

def _zone_for_bbox(
    bbox: tuple, zones: List[Zone]
) -> Optional[Zone]:
    """Return the first active zone whose polygon contains bbox's bottom-centre."""
    if not zones:
        return None

    x1, y1, x2, y2 = bbox
    bottom_centre = ((x1 + x2) / 2.0, float(y2))

    for z in zones:
        poly_np = np.array(z.polygon, dtype=np.int32)
        if cv2.pointPolygonTest(poly_np, bottom_centre, False) >= 0:
            return z
    return None
