"""
ParkMod – src/alpr.py
Automatic License Plate Recognition (ALPR) Pipeline.

Two-stage architecture:
  Stage 1 – Plate localisation
    Preferred : dedicated YOLOv8 plate-detector  (plate_detector.pt)
    Fallback  : heuristic crop (bottom-third of vehicle bbox,
                then edge-density sub-region search)

  Stage 2 – OCR + Validation
    EasyOCR reads the isolated plate crop.
    Every reading is stored as a candidate with a composite score.
    The best plate is selected via confidence + format + consensus
    (see plate_validator.choose_best_plate).

Key design decisions:
  • UNKNOWN is *never* permanently locked in.
    Every call to read_plate() can add better candidates.
  • Candidates are stored per vehicle-ID with timestamps so
    the system can reason about freshness.
  • High-confidence plate crops are saved to disk for evidence.
  • A dedicated preprocessing step (grayscale → CLAHE → threshold)
    improves OCR accuracy on noisy / low-contrast plates.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.config import (
    OCR_GPU,
    OCR_LANGUAGES,
    OCR_MIN_CONF,
    PLATE_CROPS_DIR,
    PLATE_MODEL_PATH,
)
from src.ocr import OCRCacheManager, OCRCandidate

# ── Optional imports ──────────────────────────────────────────

try:
    import easyocr as _easyocr

    _HAS_EASYOCR = True
except ImportError:
    _HAS_EASYOCR = False

try:
    from ultralytics import YOLO as _YOLO

    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

try:
    import torch as _torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════
# ALPR Pipeline
# ═══════════════════════════════════════════════════════════════

class ALPR:
    """
    Two-stage Automatic License Plate Recognition.

    Usage:
        alpr = ALPR()
        plate = alpr.read_plate(frame, bbox, vehicle_id)
        best  = alpr.get_best_plate(vehicle_id)
    """

    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self._ocr_cache = OCRCacheManager(min_agreement=2)

        # Models
        self._plate_detector = None
        self._plate_detector_backend = "none"
        self._ocr_reader = None

        # Internal frame counter (for candidate metadata)
        self._frame_idx = 0

        if not demo_mode:
            self._init_models()

    # ── Model initialisation ──────────────────────────────────

    def _init_models(self) -> None:
        # EasyOCR
        if _HAS_EASYOCR:
            try:
                self._ocr_reader = _easyocr.Reader(
                    OCR_LANGUAGES, gpu=OCR_GPU
                )
                print("[ALPR] EasyOCR initialised.")
            except Exception as exc:
                print(f"[ALPR] EasyOCR init failed: {exc}")
        else:
            print("[ALPR] EasyOCR not installed – OCR disabled.")

        # Plate detector (optional dedicated YOLO)
        plate_path = Path(PLATE_MODEL_PATH)
        if _HAS_YOLO and plate_path.exists():
            # This checkpoint is from legacy YOLOv5 training, so prefer
            # compatibility loading first to avoid Ultralytics auto-install noise.
            if (
                plate_path.name.lower() == "plate_detector.pt"
                and self._try_load_legacy_plate_detector(plate_path)
            ):
                print(
                    "[ALPR] Plate detector loaded via YOLOv5 "
                    "legacy compatibility backend."
                )
                return

            try:
                self._plate_detector = _YOLO(str(plate_path))
                self._plate_detector_backend = "ultralytics"
                print(f"[ALPR] Plate detector loaded: {plate_path.name}")
            except Exception as exc:
                print(f"[ALPR] Plate detector load failed: {exc}")
                if self._try_load_legacy_plate_detector(plate_path):
                    print(
                        "[ALPR] Plate detector loaded via YOLOv5 "
                        "legacy compatibility backend."
                    )
                else:
                    print(
                        "[ALPR] Legacy compatibility load failed → "
                        "using heuristic plate localisation."
                    )
        else:
            print(
                "[ALPR] No plate_detector.pt found → "
                "using heuristic plate localisation."
            )

    def _try_load_legacy_plate_detector(self, plate_path: Path) -> bool:
        """Load legacy YOLOv5 checkpoints that require models.yolo modules."""
        if not _HAS_TORCH:
            return False

        candidate_repos = [
            Path("models/yolov5_legacy_runtime"),
            Path("models/Automatic-Number-Plate-Recognition-using-YOLOv5-main"),
        ]

        for repo in candidate_repos:
            repo_path = repo.resolve()
            if not repo_path.exists() or not (repo_path / "hubconf.py").exists():
                continue
            try:
                os.environ.setdefault("YOLOv5_AUTOINSTALL", "False")

                # Clear conflicting cached modules from previous failed loads.
                for mod_name in (
                    "models",
                    "models.yolo",
                    "models.common",
                    "utils",
                    "utils.general",
                    "utils.dataloaders",
                ):
                    sys.modules.pop(mod_name, None)

                if str(repo_path) not in sys.path:
                    sys.path.insert(0, str(repo_path))

                self._plate_detector = _torch.hub.load(
                    str(repo_path),
                    "custom",
                    path=str(plate_path.resolve()),
                    source="local",
                    force_reload=False,
                )
                self._plate_detector_backend = "yolov5-legacy"
                return True
            except Exception:
                continue

        return False

    def is_plate_detector_ready(self) -> bool:
        """Public health-check used by dashboard warnings."""
        return self._plate_detector is not None

    def get_best_plate(self, vehicle_id: int) -> str:
        """Return the current best plate for a vehicle (never locks UNKNOWN)."""
        return self._ocr_cache.get_best_plate(vehicle_id)

    def get_candidates(self, vehicle_id: int) -> List[OCRCandidate]:
        return self._ocr_cache.get_candidates(vehicle_id)

    # ── Main entry point ──────────────────────────────────────

    def read_plate(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple,
        vehicle_id: int,
    ) -> str:
        """
        Run the full 2-stage ALPR on one vehicle.

        1. Localise the plate region inside the vehicle bbox.
        2. Preprocess the plate crop for OCR.
        3. Run EasyOCR.
        4. Clean, validate, score each reading.
        5. Store as candidates; return current best plate.

        This can be called every frame – it accumulates evidence
        and NEVER permanently locks in a bad read.
        """
        self._frame_idx += 1

        if self.demo_mode:
            return self._demo_plate(vehicle_id)

        # ── Stage 1: plate localisation ───────────────────────
        plate_crop = self._localise_plate(frame, vehicle_bbox)
        if plate_crop is None or plate_crop.size == 0:
            return self.get_best_plate(vehicle_id)

        # ── Stage 2: OCR ──────────────────────────────────────
        if self._ocr_reader is None:
            return self.get_best_plate(vehicle_id)

        # Multi-pass OCR for stronger candidate diversity
        processed = self._preprocess_plate(plate_crop)
        variants = [processed]
        if len(processed.shape) == 2:
            variants.append(cv2.bitwise_not(processed))

        parsed_results = []
        max_conf_seen = 0.0

        for variant in variants:
            try:
                results = self._ocr_reader.readtext(variant)
            except Exception:
                continue

            for _, raw_text, conf in results:
                if conf < OCR_MIN_CONF:
                    continue
                parsed_results.append((raw_text, float(conf)))
                if conf > max_conf_seen:
                    max_conf_seen = float(conf)

        best = self._ocr_cache.update(
            vehicle_id=vehicle_id,
            ocr_results=parsed_results,
            frame_index=self._frame_idx,
        )

        # Save evidence crop only when OCR had a strong confident read.
        if best != "UNKNOWN" and max_conf_seen >= max(0.65, OCR_MIN_CONF):
            self._save_plate_crop(plate_crop, vehicle_id, best)

        return best

    # ── Stage 1: plate localisation ───────────────────────────

    def _localise_plate(
        self, frame: np.ndarray, vehicle_bbox: tuple
    ) -> Optional[np.ndarray]:
        """
        Extract the license-plate region from the vehicle.

        Preferred : YOLOv8 plate detector on vehicle crop.
        Fallback  : Heuristic crop (bottom region + edge analysis).
        """
        x1, y1, x2, y2 = map(int, vehicle_bbox)

        # Clamp to frame boundaries
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return None

        # ── Try dedicated plate detector first ────────────────
        if self._plate_detector is not None:
            crop = self._detect_plate_model(vehicle_crop)
            if crop is not None:
                return crop

        # ── Heuristic fallback ────────────────────────────────
        return self._detect_plate_heuristic(vehicle_crop)

    def _detect_plate_model(
        self, vehicle_crop: np.ndarray
    ) -> Optional[np.ndarray]:
        """Use whichever plate-detector backend is loaded."""
        try:
            if self._plate_detector_backend == "ultralytics":
                results = self._plate_detector(vehicle_crop, verbose=False)
                boxes = results[0].boxes
                if len(boxes) == 0:
                    return None

                best_idx = boxes.conf.argmax()
                bx1, by1, bx2, by2 = map(
                    int, boxes.xyxy[best_idx].tolist()
                )
                crop = vehicle_crop[by1:by2, bx1:bx2]
                return crop if crop.size > 0 else None

            if self._plate_detector_backend == "yolov5-legacy":
                preds = self._plate_detector(vehicle_crop)
                det = preds.xyxy[0]
                if det is None or len(det) == 0:
                    return None

                if hasattr(det, "cpu"):
                    det_np = det.cpu().numpy()
                else:
                    det_np = np.array(det)

                # YOLOv5 xyxy + conf + cls
                conf_col = det_np[:, 4]
                best_idx = int(np.argmax(conf_col))
                bx1, by1, bx2, by2 = map(int, det_np[best_idx, :4].tolist())

                bx1 = max(0, min(bx1, vehicle_crop.shape[1] - 1))
                bx2 = max(0, min(bx2, vehicle_crop.shape[1]))
                by1 = max(0, min(by1, vehicle_crop.shape[0] - 1))
                by2 = max(0, min(by2, vehicle_crop.shape[0]))

                crop = vehicle_crop[by1:by2, bx1:bx2]
                return crop if crop.size > 0 else None

            return None
        except Exception:
            return None

    def _detect_plate_heuristic(
        self, vehicle_crop: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Heuristic plate localisation:
          1. Crop the bottom 40% of the vehicle (plates are low).
          2. Convert to grayscale → Canny edges.
          3. Find the densest horizontal band (likely the plate).
          4. Return that sub-region.
        """
        vh, vw = vehicle_crop.shape[:2]
        if vh < 20 or vw < 20:
            return None

        # Bottom 40% of the vehicle
        bottom_start = int(vh * 0.60)
        bottom_crop = vehicle_crop[bottom_start:vh, :]

        bh, bw = bottom_crop.shape[:2]
        if bh < 10 or bw < 10:
            return bottom_crop

        # Edge detection to find plate-like rectangular region
        gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 200)

        # Sum edge density per row → find the densest horizontal band
        row_density = np.sum(edges, axis=1)
        if row_density.max() == 0:
            return bottom_crop

        # Find contiguous high-density region (plate zone)
        threshold = row_density.max() * 0.3
        active = row_density > threshold
        if not np.any(active):
            return bottom_crop

        rows = np.where(active)[0]
        top_row = max(0, rows[0] - 5)
        bot_row = min(bh, rows[-1] + 5)

        # Ensure minimum height
        if (bot_row - top_row) < 10:
            return bottom_crop

        plate_region = bottom_crop[top_row:bot_row, :]
        return plate_region if plate_region.size > 0 else bottom_crop

    # ── Plate crop preprocessing ──────────────────────────────

    @staticmethod
    def _preprocess_plate(crop: np.ndarray) -> np.ndarray:
        """
        Enhance the plate crop for better OCR accuracy.

        Steps:
          1. Resize to a standard width (300 px) for consistency.
          2. Convert to grayscale.
          3. Apply CLAHE for contrast enhancement.
          4. Light Gaussian blur to reduce noise.
          5. Adaptive threshold for clean black-on-white text.
        """
        h, w = crop.shape[:2]
        if w < 10 or h < 5:
            return crop

        # Resize to standard width
        target_w = 300
        scale = target_w / w
        resized = cv2.resize(
            crop, (target_w, int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

        # Grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Light blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Adaptive threshold → clean binary image
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        return binary

    # ── Evidence saving ───────────────────────────────────────

    @staticmethod
    def _save_plate_crop(
        crop: np.ndarray, vehicle_id: int, plate_text: str
    ) -> None:
        """Save a high-confidence plate crop to disk."""
        try:
            fname = f"VH{vehicle_id:04d}_{plate_text}.jpg"
            path = PLATE_CROPS_DIR / fname
            if not path.exists():
                cv2.imwrite(str(path), crop)
        except Exception:
            pass

    # ── Demo mode ─────────────────────────────────────────────

    _DEMO_PLATES = [
        "MH12AB1234", "DL3CAF5678", "KA09MN2345", "TN22XY9012",
        "GJ01HJ3456", "UP14BR6789", "RJ14CE4321", "HR26DQ8765",
        "WB06AC1122", "PB10EF3344", "MP09GH5566", "CG04KL7788",
    ]

    def _demo_plate(self, vehicle_id: int) -> str:
        plate = self._DEMO_PLATES[vehicle_id % len(self._DEMO_PLATES)]
        return self._ocr_cache.update(
            vehicle_id=vehicle_id,
            ocr_results=[(plate, 0.99)],
            frame_index=self._frame_idx,
        )

    # ── Utilities ─────────────────────────────────────────────

    def clear_all(self) -> None:
        """Wipe all candidate histories."""
        self._ocr_cache.clear_all()

    def clear_vehicle(self, vehicle_id: int) -> None:
        """Wipe history for a single vehicle."""
        self._ocr_cache.clear_vehicle(vehicle_id)

    def summary(self) -> dict:
        """Return a quick summary of ALPR state."""
        return self._ocr_cache.summary()
