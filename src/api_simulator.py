"""
ParkMod – src/api_simulator.py
Simulates sending violation data to an enforcement authority API.
Logs the payload that would be sent in a real deployment.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from src.config import API_ENDPOINT

# Set up logger for API simulation
logger = logging.getLogger("ParkMod.API")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def build_violation_payload(
    vehicle_id: int,
    plate_number: str,
    duration_sec: float,
    location: str,
    image_path: str = "",
    evidence_url: Optional[str] = None,
) -> dict:
    """
    Build a structured JSON payload for the enforcement API.

    In a real system this would be sent as a POST request.
    """
    payload = {
        "api_version": "1.0",
        "endpoint": API_ENDPOINT,
        "timestamp": datetime.now().isoformat(),
        "violation": {
            "vehicle_id": vehicle_id,
            "plate_number": plate_number,
            "duration_sec": round(duration_sec, 2),
            "location": location,
            "status": "Violation",
            "image_path": image_path,
            "evidence_url": evidence_url
            or f"https://evidence.parkmod.local/{vehicle_id}",
        },
        "authority": {
            "name": "Municipal Parking Enforcement",
            "code": "MPE-001",
        },
    }
    return payload


def send_violation_to_api(
    vehicle_id: int,
    plate_number: str,
    duration_sec: float,
    location: str,
    image_path: str = "",
) -> dict:
    """
    SIMULATED API call – logs the violation payload.

    In production this would be an HTTP POST to the enforcement
    authority's API. Here we just build the payload, log it,
    and return the simulated response.

    Parameters
    ----------
    vehicle_id : int
    plate_number : str
    duration_sec : float
    location : str
    image_path : str

    Returns
    -------
    dict : simulated API response
    """
    payload = build_violation_payload(
        vehicle_id=vehicle_id,
        plate_number=plate_number,
        duration_sec=duration_sec,
        location=location,
        image_path=image_path,
    )

    # ── Simulate the POST request ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("📡 SIMULATED API CALL → %s", API_ENDPOINT)
    logger.info("Payload:\n%s", json.dumps(payload, indent=2))
    logger.info("=" * 60)

    # ── Simulated response ─────────────────────────────────────────────────
    response = {
        "status_code": 200,
        "message": "Violation recorded successfully (SIMULATED)",
        "reference_id": f"REF-{vehicle_id:04d}-{datetime.now().strftime('%H%M%S')}",
        "payload": payload,
    }

    logger.info(
        "✅ API Response: %s [ref: %s]",
        response["message"],
        response["reference_id"],
    )

    return response


def batch_send(records: list, location: str) -> list:
    """
    Send multiple violation records to the API (simulated).

    Parameters
    ----------
    records : list of dict
        Each dict must have: vehicle_id, plate_number, duration_sec.
    location : str

    Returns
    -------
    list of simulated API responses.
    """
    responses = []
    for rec in records:
        resp = send_violation_to_api(
            vehicle_id=rec.get("vehicle_id", 0),
            plate_number=rec.get("plate_number", "UNKNOWN"),
            duration_sec=rec.get("duration_sec", 0),
            location=location,
            image_path=rec.get("image_path", ""),
        )
        responses.append(resp)
    return responses
