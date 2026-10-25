"""
Central AI-Tails configuration (read from environment variables / .env file).
See .env.example for the full list.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Vision ------------------------------------------------------------------
# YOLOv8n weights for cat face detection (provided by AI-Tails, not versioned in git)
YOLO_WEIGHTS = Path(os.getenv("YOLO_WEIGHTS", PROJECT_ROOT / "models" / "cat_face_yolov8n.pt"))
YOLO_MIN_CONF = float(os.getenv("YOLO_MIN_CONF", "0.35"))

# Tech4Animals API (Martvel et al., IJCV 2024) for the 48 facial landmarks
LANDMARKS_API_URL = os.getenv("LANDMARKS_API_URL", "")
LANDMARKS_API_TIMEOUT = float(os.getenv("LANDMARKS_API_TIMEOUT", "30"))

# --- Thermal -----------------------------------------------------------------
# Surface->body offset measured on the TICD dataset (64 healthy indoor cats:
# top-3 reading at 32x24 has a median of 30.8 °C vs. a normal body temperature of 38.6 °C).
THERMAL_BODY_OFFSET_C = float(os.getenv("THERMAL_BODY_OFFSET_C", "7.8"))

# --- Azure OpenAI ------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")


def azure_configured() -> bool:
    return bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT)
