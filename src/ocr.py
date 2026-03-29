"""
ParkMod – src/ocr.py
OCR candidate cache and update logic for ALPR.

This module does not run OCR directly. It stores and aggregates
OCR readings produced elsewhere (for example by src/alpr.py),
then selects the best plate per vehicle based on confidence,
agreement across repeated reads, and Indian plate validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from src.plate_validator import (
    extract_plate_candidates,
    is_indian_plate_like,
    is_valid_indian_plate,
    score_plate_candidate,
    select_best_candidate,
)


@dataclass
class OCRCandidate:
    """One cleaned OCR reading for a tracked vehicle."""

    text: str
    confidence: float
    score: float
    frame_index: int
    timestamp: float
    valid_indian: bool


@dataclass
class OCRVehicleCache:
    """Per-vehicle OCR memory with multiple candidate observations."""

    vehicle_id: int
    candidates: List[OCRCandidate] = field(default_factory=list)
    best_plate: str = "UNKNOWN"

    def add(self, candidate: OCRCandidate) -> None:
        self.candidates.append(candidate)


class OCRCacheManager:
    """
    Stores and updates OCR candidates per vehicle.

    UNKNOWN is never cached as a final lock; if no new valid reads are
    present, the previous best plate is retained.
    """

    def __init__(self, min_agreement: int = 2):
        self._cache: Dict[int, OCRVehicleCache] = {}
        self._min_agreement = min_agreement

    def _get(self, vehicle_id: int) -> OCRVehicleCache:
        if vehicle_id not in self._cache:
            self._cache[vehicle_id] = OCRVehicleCache(vehicle_id=vehicle_id)
        return self._cache[vehicle_id]

    def update(
        self,
        vehicle_id: int,
        ocr_results: Iterable[Tuple[str, float]],
        frame_index: int,
    ) -> str:
        """
        Update candidate history from raw OCR outputs.

        Parameters
        ----------
        ocr_results : iterable of (raw_text, confidence)
            Raw OCR fragments from the recognizer.
        """
        state = self._get(vehicle_id)
        now = time.time()

        for raw_text, conf in ocr_results:
            if conf <= 0:
                continue

            for token in extract_plate_candidates(raw_text):
                if not is_indian_plate_like(token):
                    continue

                score = score_plate_candidate(token, conf)
                if score <= 0:
                    continue

                state.add(
                    OCRCandidate(
                        text=token,
                        confidence=round(conf, 3),
                        score=score,
                        frame_index=frame_index,
                        timestamp=now,
                        valid_indian=is_valid_indian_plate(token),
                    )
                )

        if not state.candidates:
            return state.best_plate

        ranked_input = [(cand.text, cand.score) for cand in state.candidates]
        selected = select_best_candidate(
            ranked_input,
            min_votes=self._min_agreement,
        )

        if selected != "UNKNOWN":
            state.best_plate = selected
        return state.best_plate

    def get_best_plate(self, vehicle_id: int) -> str:
        return self._get(vehicle_id).best_plate

    def get_candidates(self, vehicle_id: int) -> List[OCRCandidate]:
        return self._get(vehicle_id).candidates

    def clear_vehicle(self, vehicle_id: int) -> None:
        self._cache.pop(vehicle_id, None)

    def clear_all(self) -> None:
        self._cache.clear()

    def summary(self) -> dict:
        return {
            vid: {
                "best": cache.best_plate,
                "reads": len(cache.candidates),
            }
            for vid, cache in self._cache.items()
        }
