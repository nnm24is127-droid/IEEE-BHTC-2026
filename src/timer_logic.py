"""
ParkMod – src/timer_logic.py
Monitors parking duration for each tracked vehicle inside the ROI.
If a vehicle stays beyond the threshold → violation is triggered.
Timer resets when the vehicle exits the ROI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.config import DEFAULT_VIOLATION_THRESHOLD_SEC, DUPLICATE_COOLDOWN_SEC


@dataclass
class VehicleTimer:
    """Timer record for a single tracked vehicle."""

    vehicle_id: int
    entry_time: float  # time.time() when entered ROI
    is_in_roi: bool = True
    duration_sec: float = 0.0
    is_violation: bool = False
    violation_time: Optional[float] = (
        None  # timestamp when violation was flagged
    )


class ParkingTimerManager:
    """
    Manages entry/exit timers for vehicles inside the ROI.

    Features:
    - Stores entry time when vehicle enters ROI
    - Computes real-time duration
    - Triggers violation when duration > threshold
    - Resets timer when vehicle exits ROI
    - Prevents duplicate violation for same vehicle within cooldown

    Parameters
    ----------
    threshold_sec : float
        Seconds a vehicle must stay in ROI to trigger violation.
    cooldown_sec : float
        Seconds before the same vehicle can re-trigger a violation.
    """

    def __init__(
        self,
        threshold_sec: float = DEFAULT_VIOLATION_THRESHOLD_SEC,
        cooldown_sec: float = DUPLICATE_COOLDOWN_SEC,
    ):
        self.threshold_sec = threshold_sec
        self.cooldown_sec = cooldown_sec
        self._timers: Dict[int, VehicleTimer] = {}
        self._violation_log: Dict[
            int, float
        ] = {}  # vehicle_id → last violation time

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        active_ids_in_roi: List[int],
        all_tracked_ids: List[int],
        current_time_sec: Optional[float] = None,
    ) -> Tuple[List[int], Dict[int, VehicleTimer]]:
        """
        Call every frame with:
        - active_ids_in_roi : vehicle IDs currently inside ROI
        - all_tracked_ids   : all vehicle IDs being tracked
        - current_time_sec  : explicit time for video simulation, or None for real-time

        Returns
        -------
        new_violations : list of vehicle IDs that JUST violated this frame
        timers         : dict of all active VehicleTimer records
        """
        if current_time_sec is None:
            current_time_sec = time.time()
        now = current_time_sec
        new_violations: List[int] = []

        # ── Vehicles that entered / remain in ROI ──────────────────────────
        for vid in active_ids_in_roi:
            if vid not in self._timers:
                # Vehicle just entered ROI → start timer
                self._timers[vid] = VehicleTimer(
                    vehicle_id=vid,
                    entry_time=now,
                    is_in_roi=True,
                )
            else:
                self._timers[vid].is_in_roi = True

            timer = self._timers[vid]
            timer.duration_sec = now - timer.entry_time

            # Check violation threshold
            if (
                timer.duration_sec >= self.threshold_sec
                and not timer.is_violation
            ):
                # Check cooldown (avoid duplicate violations)
                last_viol = self._violation_log.get(vid)
                if last_viol is None or (now - last_viol) > self.cooldown_sec:
                    timer.is_violation = True
                    timer.violation_time = now
                    self._violation_log[vid] = now
                    new_violations.append(vid)

        # ── Vehicles that exited ROI → reset timer ─────────────────────────
        for vid in list(self._timers.keys()):
            if vid not in active_ids_in_roi:
                self._timers[vid].is_in_roi = False
                # If vehicle left ROI, reset timer fully
                if vid in all_tracked_ids:
                    # Still tracked but outside ROI → reset entry
                    del self._timers[vid]
                else:
                    # Lost tracking → clean up
                    del self._timers[vid]

        return new_violations, dict(self._timers)

    def get_timer(self, vehicle_id: int) -> Optional[VehicleTimer]:
        return self._timers.get(vehicle_id)

    def get_all_timers(self) -> Dict[int, VehicleTimer]:
        return dict(self._timers)

    def get_duration(self, vehicle_id: int) -> float:
        """Return current duration in seconds for a vehicle, or 0."""
        t = self._timers.get(vehicle_id)
        if t:
            return t.duration_sec
        return 0.0

    def get_status(self, vehicle_id: int) -> str:
        """Return status string: Violation / Warning / Compliant / Outside."""
        t = self._timers.get(vehicle_id)
        if not t:
            return "Outside"
        if t.is_violation:
            return "Violation"
        if t.duration_sec >= self.threshold_sec * 0.5:
            return "Warning"
        return "Compliant"

    def reset(self) -> None:
        """Clear all timers."""
        self._timers.clear()
        self._violation_log.clear()
