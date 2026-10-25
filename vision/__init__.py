"""
Vision module for AI-Tails: cat face detection (YOLO), FGS landmarks (t4a API) and thermal analysis.
"""

from .detector import (
    DEFAULT_LANDMARK_MAP,
    LandmarksApiError,
    analyze_landmarks_detailed,
    calc_distance,
    extract_fgs_scores_from_landmarks,
    generate_mock_landmarks,
    send_image_for_processing,
)
from .face import FaceModelMissing, detect_cat_face
from .thermal_extractor import ThermalImageError, analyze_thermal_image, classify_temperature

__all__ = [
    "DEFAULT_LANDMARK_MAP", "LandmarksApiError", "analyze_landmarks_detailed", "calc_distance",
    "extract_fgs_scores_from_landmarks", "generate_mock_landmarks", "send_image_for_processing",
    "FaceModelMissing", "detect_cat_face",
    "ThermalImageError", "analyze_thermal_image", "classify_temperature",
]
