"""
Modulo Vision per AI-Tails.
Esporta le funzioni di estrazione e calcolo geometrico dei landmark.
"""

from .detector import (
    calc_distance,
    extract_fgs_scores_from_landmarks,
    analyze_landmarks_detailed,
    generate_landmarks_for_image,
    send_image_for_processing,
    generate_mock_landmarks,
    DEFAULT_LANDMARK_MAP
)

__all__ = [
    "calc_distance",
    "extract_fgs_scores_from_landmarks",
    "analyze_landmarks_detailed",
    "generate_landmarks_for_image",
    "send_image_for_processing",
    "generate_mock_landmarks",
    "DEFAULT_LANDMARK_MAP"
]
