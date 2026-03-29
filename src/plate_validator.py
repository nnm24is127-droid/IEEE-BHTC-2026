"""
ParkMod – src/plate_validator.py
Indian license-plate text cleaning, validation, scoring, and selection.

Covers:
- Text cleaning  (strip noise characters, fix common OCR confusions)
- Format validation  (strict Indian RTO patterns)
- Noise rejection  (brand names, common garbage words)
- Candidate scoring  (combines OCR confidence + format match + consensus)
- Best-plate selection  (picks the highest-scored valid candidate)

Indian plate format reference:
  [StateCode 2A] [RTO 1-2D] [Series 0-3A] [Number 1-4D]
  Examples: KA01AB1234, MH12DE1433, DL5CAB1234, TN22XY9012
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# All valid Indian state / UT codes  (2 letters)
_VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

# Brand names / noise words the OCR commonly confuses for plates
_NOISE_WORDS = {
    "HONDA", "SUZUKI", "TOYOTA", "MARUTI", "HYUNDAI", "MAHINDRA",
    "TATA", "FORD", "BMW", "AUDI", "MERCEDES", "KIA", "NISSAN",
    "RENAULT", "ŠKODA", "SKODA", "VOLKSWAGEN", "CHEVROLET",
    "INDIA", "DIESEL", "PETROL", "TURBO", "SPORT", "EDITION",
    "AUTOMATIC", "MANUAL", "HYBRID", "ELECTRIC", "MARUTISUZUKI",
    "SHIFTGEAR", "SMARTPLAY", "NEXON", "BALENO", "SWIFT", "DZIRE",
    "CRETA", "VERNA", "CITY", "AMAZE", "INNOVA", "FORTUNER",
    "BREZZA", "ERTIGA", "ALTO", "WAGONR", "WAGON", "SCORPIO",
    "BOLERO", "THAR", "HARRIER", "SAFARI", "PUNCH", "VENUE",
    "SELTOS", "SONET", "CARNIVAL", "CARENS", "XUVI", "XUV",
    "WARNING", "CAUTION", "DANGER", "NOTICE", "PARKING", "STOP",
    "SCHOOL", "POLICE", "AMBULANCE", "FIRE", "PRESS",
}

# ── Regex patterns ────────────────────────────────────────────

# Remove everything except A-Z and 0-9
_STRIP_RE = re.compile(r"[^A-Z0-9]")
_TOKEN_RE = re.compile(r"[A-Z0-9]{4,14}")

# Primary Indian plate pattern (strict)
#   SS  DD  AAA  DDDD
#   e.g. KA 01 AB 1234   or   DL 5 C 1234
_PLATE_STRICT = re.compile(
    r"^[A-Z]{2}"        # State code (2 letters)
    r"[0-9]{1,2}"       # RTO code (1-2 digits)
    r"[A-Z]{1,3}"       # Series (1-3 letters)
    r"[0-9]{3,4}$"      # Running number (3-4 digits)
)

# Bharat (BH) series: 22BH1234AA
_PLATE_BH = re.compile(
    r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"
)

# Looser pattern (catches partial reads that are clearly plate-like)
_PLATE_LOOSE = re.compile(
    r"^[A-Z]{2}"
    r"[0-9]{1,2}"
    r"[A-Z0-9]{3,8}$"
)


# ═══════════════════════════════════════════════════════════════
# OCR confusion fixer
# ═══════════════════════════════════════════════════════════════

# Common character confusions EasyOCR makes on Indian plates
_TO_DIGIT = str.maketrans({
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
})

_TO_ALPHA = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
})


def _alnum_tokenize(raw_text: str) -> List[str]:
    base = raw_text.upper().strip()
    tokens = _TOKEN_RE.findall(base)
    merged = _STRIP_RE.sub("", base)
    if len(merged) >= 4:
        tokens.append(merged)
    unique: List[str] = []
    for t in tokens:
        if t not in unique:
            unique.append(t)
    return unique


def _coerce_indian_candidate(text: str) -> str:
    """Coerce likely OCR confusions into a canonical Indian-plate-like token."""
    if len(text) < 6:
        return text

    token = text

    # Special handling for BH series: 22BH1234AA
    if "BH" in token and len(token) >= 8:
        out = list(token)
        # Year block
        out[0] = out[0].translate(_TO_DIGIT)
        out[1] = out[1].translate(_TO_DIGIT)
        # Running number block
        for i in range(4, min(8, len(out))):
            out[i] = out[i].translate(_TO_DIGIT)
        # Suffix letters
        for i in range(8, len(out)):
            out[i] = out[i].translate(_TO_ALPHA)
        return "".join(out)

    # Classic format coercion
    out = list(token)

    # First two chars should be letters (state code)
    out[0] = out[0].translate(_TO_ALPHA)
    out[1] = out[1].translate(_TO_ALPHA)

    # Last 3-4 chars should be digits
    tail_len = 4 if len(out) >= 9 else 3
    tail_start = max(0, len(out) - tail_len)
    for i in range(tail_start, len(out)):
        out[i] = out[i].translate(_TO_DIGIT)

    # RTO block near the front should be digits
    if len(out) >= 4:
        out[2] = out[2].translate(_TO_DIGIT)
        out[3] = out[3].translate(_TO_DIGIT)

    # Middle section should be letters
    for i in range(4, tail_start):
        out[i] = out[i].translate(_TO_ALPHA)

    return "".join(out)


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def clean_plate_text(raw_text: str) -> str:
    """
    Clean raw OCR output to a normalised plate candidate.

    Steps:
      1. Uppercase + strip whitespace
      2. Remove all non-alphanumeric characters
      3. Attempt OCR confusion fixes
      4. Return empty string if result is < 4 chars
    """
    text = _STRIP_RE.sub("", raw_text.upper().strip())
    if len(text) < 4:
        return ""
    return text


def extract_plate_candidates(raw_text: str) -> List[str]:
    """Extract and normalise plate-like tokens from a raw OCR string."""
    out: List[str] = []
    for token in _alnum_tokenize(raw_text):
        cleaned = clean_plate_text(token)
        if not cleaned:
            continue
        coerced = _coerce_indian_candidate(cleaned)
        if coerced not in out:
            out.append(coerced)
    return out


def is_noise_word(text: str) -> bool:
    """Return True if the text matches a known brand / noise word."""
    return text in _NOISE_WORDS or text.rstrip("0123456789") in _NOISE_WORDS


def is_valid_indian_plate(text: str) -> bool:
    """
    Return True if text matches a strict Indian plate format
    AND the state code is a real Indian state/UT.
    """
    if _PLATE_BH.match(text):
        return True

    if _PLATE_STRICT.match(text):
        state = text[:2]
        return state in _VALID_STATE_CODES

    return False


def is_indian_plate_like(text: str) -> bool:
    """
    Looser check – the text *looks* plate-like even if it
    doesn't perfectly match the strict format (partial reads).
    """
    if len(text) < 7 or len(text) > 12:
        return False
    if is_noise_word(text):
        return False
    if _PLATE_BH.match(text):
        return True
    # Allow near-miss state codes in loose mode for OCR tolerance.
    # Strict validation still requires real state/UT codes.
    return bool(_PLATE_LOOSE.match(text))


def is_plausible_plate(text: str) -> bool:
    """Backward-compatible alias for Indian plate-like validation."""
    return is_indian_plate_like(text)


def score_plate_candidate(text: str, ocr_confidence: float) -> float:
    """
    Produce a composite score for a plate candidate.

    Scoring breakdown:
      - base            = ocr_confidence  (0.0 – 1.0)
      - strict match    → +3.0  (very strong boost)
      - plausible match → +1.0  (moderate boost)
      - noise word      → -5.0  (kill it)
      - too short/long  → -1.0
    """
    score = ocr_confidence

    if is_noise_word(text):
        return -5.0                      # Instant reject

    if is_valid_indian_plate(text):
        score += 3.0                     # Strict/accepted Indian format
    elif is_indian_plate_like(text):
        score += 0.5                     # Looks Indian-plate-like but partial
    else:
        if len(text) < 7 or len(text) > 12:
            score -= 1.0                 # Unlikely plate length

    return round(score, 3)


def choose_best_plate(
    candidates: List[Tuple[str, float]],
    min_agreement: int = 2,
) -> str:
    """
    Select the best plate from a list of (text, score) candidates.

    Strategy:
      1. Filter out negative-scored candidates (noise).
      2. If any candidate appears ≥ min_agreement times AND is
         a valid Indian plate, prefer it (consensus).
      3. Otherwise fall back to the single highest-scored candidate.
      4. Return "UNKNOWN" only if no viable candidate exists.
    """
    if not candidates:
        return "UNKNOWN"

    # Step 1: remove garbage
    viable = [(t, s) for t, s in candidates if s > 0]
    if not viable:
        return "UNKNOWN"

    # Step 2: consensus among valid plates
    valid_texts = [t for t, s in viable if is_valid_indian_plate(t)]
    if valid_texts:
        counts = Counter(valid_texts)
        # Pick the plate that appeared most often
        most_common, freq = counts.most_common(1)[0]
        if freq >= min_agreement:
            return most_common
        # Even without consensus, the best-scored valid plate wins
        valid_scored = [(t, s) for t, s in viable
                        if is_valid_indian_plate(t)]
        valid_scored.sort(key=lambda x: x[1], reverse=True)
        return valid_scored[0][0]

    # Step 3: no strictly valid plate → highest score among plausible
    plausible = [(t, s) for t, s in viable if is_indian_plate_like(t)]
    if plausible:
        plausible.sort(key=lambda x: x[1], reverse=True)
        return plausible[0][0]

    # Step 4: truly nothing usable
    return "UNKNOWN"


def aggregate_candidate_scores(
    candidates: List[Tuple[str, float]],
) -> List[Tuple[str, float, int]]:
    """Aggregate repeated OCR candidates into weighted final scores."""
    buckets: dict[str, List[float]] = {}
    for text, score in candidates:
        if score <= 0:
            continue
        if not is_indian_plate_like(text):
            continue
        buckets.setdefault(text, []).append(score)

    aggregated: List[Tuple[str, float, int]] = []
    for text, scores in buckets.items():
        count = len(scores)
        best = max(scores)
        mean = sum(scores) / count
        agreement_bonus = 0.45 * (count - 1)
        strict_bonus = 0.75 if is_valid_indian_plate(text) else 0.0
        final = best + (0.25 * mean) + agreement_bonus + strict_bonus
        aggregated.append((text, round(final, 3), count))

    aggregated.sort(key=lambda item: item[1], reverse=True)
    return aggregated


def select_best_candidate(
    candidates: List[Tuple[str, float]],
    min_votes: int = 2,
) -> str:
    """
    Pick best candidate using confidence and repeated agreement.

    Selection priority:
      1. Strictly valid Indian plate with >= min_votes.
      2. Highest aggregated strict valid Indian plate.
      3. Highest aggregated Indian-plate-like candidate.
      4. UNKNOWN.
    """
    ranked = aggregate_candidate_scores(candidates)
    if not ranked:
        return "UNKNOWN"

    voted_valid = [
        (text, score, cnt)
        for text, score, cnt in ranked
        if cnt >= min_votes and is_valid_indian_plate(text)
    ]
    if voted_valid:
        voted_valid.sort(key=lambda item: item[1], reverse=True)
        return voted_valid[0][0]

    strict = [row for row in ranked if is_valid_indian_plate(row[0])]
    if strict:
        return strict[0][0]

    return ranked[0][0]
