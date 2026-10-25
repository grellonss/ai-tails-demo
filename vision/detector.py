"""
Vision module for AI-Tails: geometric bridge between the 48 facial landmarks (Tech4Animals API)
and the 3 Feline Grimace Scale action units used by the ontology.
"""

import base64
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

Point = Union[Dict[str, float], Sequence[float]]

# =============================================================================
# 1. GEOMETRY UTILITIES
# =============================================================================

def _xy(p: Point) -> Tuple[float, float]:
    if isinstance(p, dict) and "x" in p and "y" in p:
        return p["x"], p["y"]
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return p[0], p[1]
    raise ValueError(f"Invalid point format: {p}")


def calc_distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points {'x','y'} or (x, y)."""
    return math.dist(_xy(p1), _xy(p2))


# =============================================================================
# 2. LANDMARK INDEX MAP (to be calibrated on the 48 points of the t4a API)
# =============================================================================
# NOTE FOR THE TEAM: indices 0-47 are the 48 points of the t4a API (CatFLW layout).
# If you verify different indices on real images, update only this dictionary.
DEFAULT_LANDMARK_MAP = {
    # Eye (Eye Aspect Ratio): height vs width
    "eye_top": 10, "eye_bottom": 14, "eye_inner": 8, "eye_outer": 12,
    # Ears: distance between the tips relative to skull width
    "ear_tip_left": 2, "ear_tip_right": 40, "head_left": 0, "head_right": 25,
    # Muzzle: height / width (round vs elliptical/tense)
    "muzzle_top": 30, "muzzle_bottom": 35, "muzzle_left": 32, "muzzle_right": 37,
}

# =============================================================================
# 3. GEOMETRIC BRIDGE: ratios -> FGS scores (0 absent, 1 moderate, 2 marked)
# =============================================================================
# For each action unit: (numerator, denominator) of the ratio, thresholds and descriptions.
# "low_is_pain": for eyes and muzzle a LOWER ratio means pain, for the ears a HIGHER one.
ACTION_UNITS = {
    "ear": {
        "ratio": (("ear_tip_left", "ear_tip_right"), ("head_left", "head_right")),
        "low_is_pain": False, "moderate": 1.2, "marked": 1.5,
        "labels": ("Ears upright (Absent)", "Ears slightly apart (Moderate)",
                   "Ears flattened / rotated (Marked)"),
    },
    "eye": {
        "ratio": (("eye_top", "eye_bottom"), ("eye_inner", "eye_outer")),
        "low_is_pain": True, "moderate": 0.6, "marked": 0.4,
        "labels": ("Eye wide open (Absent)", "Eye partially closed (Moderate)",
                   "Eye squinted (Marked)"),
    },
    "muzzle": {
        "ratio": (("muzzle_top", "muzzle_bottom"), ("muzzle_left", "muzzle_right")),
        "low_is_pain": True, "moderate": 0.8, "marked": 0.6,
        "labels": ("Relaxed / Round (Absent)", "Mild tension (Moderate)",
                   "Tense / Elliptical (Marked)"),
    },
}


def _score(ratio: float, au: dict) -> int:
    if au["low_is_pain"]:
        return 0 if ratio > au["moderate"] else (1 if ratio >= au["marked"] else 2)
    return 0 if ratio < au["moderate"] else (1 if ratio <= au["marked"] else 2)


def analyze_landmarks_detailed(landmarks: Union[List[Point], Dict[int, Point]],
                               indices_map: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Computes geometric ratios, discrete scores (0/1/2) and descriptions for the 3 action units.

    :return: {"scores": {ear, eye, muzzle, total}, "ratios": {...}, "descriptions": {...}}
    """
    m = indices_map or DEFAULT_LANDMARK_MAP
    scores, ratios, descriptions = {}, {}, {}
    for name, au in ACTION_UNITS.items():
        (a, b), (c, d) = au["ratio"]
        den = calc_distance(landmarks[m[c]], landmarks[m[d]])
        ratio = round(calc_distance(landmarks[m[a]], landmarks[m[b]]) / den, 3) if den > 0 else 0.0
        scores[name] = _score(ratio, au)
        ratios[f"{name}_ratio"] = ratio
        descriptions[name] = au["labels"][scores[name]]
    scores["total"] = scores["ear"] + scores["eye"] + scores["muzzle"]
    return {"scores": scores, "ratios": ratios, "descriptions": descriptions}


def extract_fgs_scores_from_landmarks(landmarks, indices_map=None) -> Tuple[int, int, int]:
    """Compatibility: returns only (ear_score, eye_score, muzzle_score)."""
    s = analyze_landmarks_detailed(landmarks, indices_map)["scores"]
    return s["ear"], s["eye"], s["muzzle"]


# =============================================================================
# 4. T4A LANDMARKS API CLIENT (Tech4Animals notebook format)
# =============================================================================

class LandmarksApiError(RuntimeError):
    """The landmarks API is not configured, does not respond or returns invalid data."""


def _inside_ratio(points: List[dict], box: dict) -> float:
    inside = [box["x1"] <= p["x"] <= box["x2"] and box["y1"] <= p["y"] <= box["y2"] for p in points]
    return sum(inside) / len(inside)


def send_image_for_processing(image_bytes: bytes, api_url: str, name: str = "image.jpg",
                              face_box: Optional[dict] = None, meta: Optional[dict] = None) -> List[Dict[str, float]]:
    """
    Sends the photo to the t4a API: POST {"name": ..., "image": "data:image/jpeg;base64,..."}.
    Expected response: [{"<animal>": {"landmarks": [{"x":..,"y":..}, ...48]}}, ...].
    If the API returns several animals, picks the one whose points fall inside the YOLO box.

    If ``meta`` is a dict, ``source`` is written into it: "api" or "mock:<profile>" (development server).

    :raises LandmarksApiError: if the URL is missing, the API does not respond or there are not 48 points
    """
    if not api_url:
        raise LandmarksApiError("LANDMARKS_API_URL is not set (see .env.example).")
    payload = {"name": name, "image": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()}
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        raise LandmarksApiError(f"Landmarks API unreachable ({api_url}): {type(e).__name__}") from e

    candidates = []
    for animal in data if isinstance(data, list) else [data]:
        for details in (animal.values() if isinstance(animal, dict) else []):
            pts = details.get("landmarks") if isinstance(details, dict) else None
            if pts and len(pts) == 48:
                candidates.append([{"x": float(p["x"]), "y": float(p["y"])} for p in pts])
    if not candidates:
        raise LandmarksApiError("The API did not return 48 landmarks for this cat.")
    if meta is not None:
        mock = response.headers.get("X-Mock-Landmarks")
        meta["source"] = f"mock:{mock}" if mock else "api"
    if face_box and len(candidates) > 1:
        candidates.sort(key=lambda pts: _inside_ratio(pts, face_box), reverse=True)
    return candidates[0]


# =============================================================================
# 5. SYNTHETIC LANDMARKS (only for tests and the development mock server)
# =============================================================================

def generate_mock_landmarks(profile: str = "healthy", offset: Tuple[float, float] = (0.0, 0.0),
                            scale: float = 1.0) -> List[Dict[str, float]]:
    """
    48 synthetic landmarks with known scores: 'healthy' (0/0/0), 'moderate' (1/1/1), 'pain' (2/2/2).
    NOT used by the app in production: they serve the tests and tools/mock_landmarks_server.py.
    """
    m = DEFAULT_LANDMARK_MAP
    pts = {i: (100.0, 100.0) for i in range(48)}
    pts[m["head_left"]], pts[m["head_right"]] = (50, 150), (250, 150)      # head 200 wide (cheeks)
    pts[m["eye_inner"]], pts[m["eye_outer"]] = (100, 130), (130, 130)      # eye 30 wide
    pts[m["muzzle_left"]], pts[m["muzzle_right"]] = (125, 185), (175, 185)  # muzzle 50 wide
    eye_h, ear_w, muzzle_h = {"healthy": (22, 200, 45), "moderate": (15, 260, 35)}.get(profile, (9, 320, 25))
    pts[m["eye_top"]], pts[m["eye_bottom"]] = (115, 130 - eye_h / 2), (115, 130 + eye_h / 2)
    pts[m["ear_tip_left"]], pts[m["ear_tip_right"]] = (150 - ear_w / 2, 40), (150 + ear_w / 2, 40)
    pts[m["muzzle_top"]], pts[m["muzzle_bottom"]] = (150, 185 - muzzle_h / 2), (150, 185 + muzzle_h / 2)
    ox, oy = offset
    return [{"x": ox + pts[i][0] * scale, "y": oy + pts[i][1] * scale} for i in range(48)]
