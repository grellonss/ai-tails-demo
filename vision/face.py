"""
Cat face detection with the YOLOv8n model provided by AI-Tails.
"""

from functools import lru_cache
from typing import Optional

import numpy as np

import config


class FaceModelMissing(RuntimeError):
    """The YOLO weights are not at the configured path."""


@lru_cache(maxsize=1)
def _model():
    if not config.YOLO_WEIGHTS.exists():
        raise FaceModelMissing(
            f"YOLO weights not found at {config.YOLO_WEIGHTS}. "
            "Copy cat_face_yolov8n.pt into the models/ folder (see README)."
        )
    from ultralytics import YOLO
    return YOLO(str(config.YOLO_WEIGHTS))


def detect_cat_face(image_bgr: np.ndarray) -> Optional[dict]:
    """
    Returns the face box with the highest confidence, or None if no face
    exceeds YOLO_MIN_CONF.

    :return: {"box": {"x1", "y1", "x2", "y2"}, "confidence": float}
    """
    result = _model()(image_bgr, verbose=False, conf=config.YOLO_MIN_CONF)[0]
    if not len(result.boxes):
        return None
    best = int(result.boxes.conf.argmax())
    x1, y1, x2, y2 = (round(float(v), 1) for v in result.boxes.xyxy[best])
    return {"box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "confidence": round(float(result.boxes.conf[best]), 3)}
