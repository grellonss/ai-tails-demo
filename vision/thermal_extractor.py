"""
Cat thermal photo analysis -> estimated body temperature -> thermal status.

Expected format: 640x480 false-colour thermal photo (or the same layout scaled) with the scale bar
on the right side and the max/min values printed at the top and bottom right, as in the images of
the TICD dataset (Abdulghafar et al., 2025) and in InfiRay/Hikmicro-style thermal cameras.

Pipeline:
  1. OCR of the scale's max/min values (digit template matching, vision/assets/thermal_digits.npz)
  2. colour of each pixel -> °C, by finding the nearest colour along the scale bar
  3. cat segmentation (Otsu threshold on temperature + largest contour)
  4. resampling to 32x24 (resolution of the station's MLX90640 sensor) and mean of the cat's
     3 hottest pixels = head surface temperature
  5. estimated body temperature = surface + offset calibrated on TICD (THERMAL_BODY_OFFSET_C)
  6. classify_temperature(): veterinary thresholds

Note: this is a rule-based classifier, not a trained model: no labelled thermal data exists to
train one. classify_temperature() keeps the signature defined by the architecture, so an ML model
can replace it without touching the rest of the pipeline.
"""

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config

# Thermal photo layout (coordinates on 640x480)
SIZE = (640, 480)
BAR_X, BAR_Y = (607, 618), (72, 408)        # colour bar: top = max, bottom = min
MAX_BOX = (36, 56, 588, 622)                # y0, y1, x0, x1 of the max value digits
MIN_BOX = (426, 446, 588, 622)              # ... and of the min value
OVERLAY_X = 545                             # right of this column there is only UI
UNIT_BOX = (415, 480, 0, 75)                # "°C" badge at the bottom left
GRID = (24, 32)                             # rows, columns of the station's thermal sensor
TOP_K = 3
GLYPH = (10, 14)
DIGITS_FILE = Path(__file__).parent / "assets" / "thermal_digits.npz"


class ThermalImageError(ValueError):
    """The thermal photo is not in the expected format (unreadable scale or no cat found)."""


# -----------------------------------------------------------------------------
# Classification (veterinary guidelines)
# -----------------------------------------------------------------------------

def classify_temperature(temp_celsius: float) -> dict:
    """
    Classifies feline body temperature:
      - < 38.0 °C        Hypothermia   (severity 1)
      - 38.0 - 39.2 °C   Normothermia  (severity 0)
      - 39.2 - 40.0 °C   MildFever     (severity 1)
      - >= 40.0 °C       HighFever     (severity 2)
    """
    if temp_celsius < 38.0:
        status, severity = "Hypothermia", 1
    elif temp_celsius <= 39.2:
        status, severity = "Normothermia", 0
    elif temp_celsius < 40.0:
        status, severity = "MildFever", 1
    else:
        status, severity = "HighFever", 2
    return {"temperature": round(temp_celsius, 1), "thermal_status": status, "thermal_score": severity}


# -----------------------------------------------------------------------------
# Reading the thermal photo
# -----------------------------------------------------------------------------

def _templates() -> dict:
    d = np.load(DIGITS_FILE)
    return {k: d[k] for k in d.files}


def _read_number(img: np.ndarray, box, templates: dict) -> Optional[int]:
    y0, y1, x0, x1 = box
    gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((gray > 140).astype(np.uint8))
    text = ""
    for _, i in sorted((stats[i, 0], i) for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= 8):
        x, y, w, h = stats[i, :4]
        glyph = cv2.resize((labels[y:y + h, x:x + w] == i).astype(np.float32), GLYPH,
                           interpolation=cv2.INTER_AREA).ravel()
        text += min(templates, key=lambda ch: np.abs(templates[ch] - glyph).sum())
    return int(text) if re.fullmatch(r"-?\d{1,2}", text) else None


def _to_celsius(img: np.ndarray, tmax: float, tmin: float) -> np.ndarray:
    """Colour -> °C via the nearest colour along the scale bar. UI -> NaN."""
    bar = img[BAR_Y[0]:BAR_Y[1], BAR_X[0]:BAR_X[1]].astype(np.float32).mean(axis=1)
    temps = np.linspace(tmax, tmin, len(bar))
    q = (img.reshape(-1, 3) // 4).astype(np.int32)                 # quantise for speed
    keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    uniq, inv = np.unique(keys, return_inverse=True)
    colors = np.stack([uniq // 4096, (uniq // 64) % 64, uniq % 64], axis=1).astype(np.float32) * 4 + 2
    nearest = ((colors[:, None, :] - bar[None, :, :]) ** 2).sum(-1).argmin(axis=1)
    t = temps[nearest][inv].reshape(img.shape[:2])
    t[:, OVERLAY_X:] = np.nan
    y0, y1, x0, x1 = UNIT_BOX
    t[y0:y1, x0:x1] = np.nan
    return t


def _segment_cat(t: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(t)
    lo, hi = np.nanmin(t), np.nanmax(t)
    u8 = np.zeros(t.shape, np.uint8)
    u8[valid] = ((t[valid] - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    _, bw = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(t.shape, np.uint8)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return mask.astype(bool) & valid


def analyze_thermal_image(image_bgr: np.ndarray) -> dict:
    """
    Analyses a thermal photo and returns the estimated temperature and thermal status.

    :return: {temperature, thermal_status, thermal_score, surface_max_c, scale: {min, max},
              hotspots: [{x, y}] (centres of the 3 hottest 32x24 pixels, in photo coordinates)}
    :raises ThermalImageError: if the scale is unreadable or the cat cannot be told apart from the background
    """
    if image_bgr is None:
        raise ThermalImageError("The thermal photo could not be read.")
    h0, w0 = image_bgr.shape[:2]
    img = cv2.resize(image_bgr, SIZE) if (w0, h0) != SIZE else image_bgr

    templates = _templates()
    tmax, tmin = _read_number(img, MAX_BOX, templates), _read_number(img, MIN_BOX, templates)
    if tmax is None or tmin is None or tmax <= tmin:
        raise ThermalImageError("Unreadable thermal photo scale (the layout with the bar on the right is required).")

    t = _to_celsius(img, tmax, tmin)
    mask = _segment_cat(t)
    filled = np.where(np.isnan(t), np.nanmedian(t), t).astype(np.float32)
    t32 = cv2.resize(filled, (GRID[1], GRID[0]), interpolation=cv2.INTER_AREA)
    m32 = cv2.resize(mask.astype(np.float32), (GRID[1], GRID[0]), interpolation=cv2.INTER_AREA) > 0.25
    if not m32.any():
        raise ThermalImageError("No cat distinguishable from the background in the thermal photo.")

    hottest = np.argsort(np.where(m32, t32, -np.inf).ravel())[::-1][:TOP_K]
    cells = [np.unravel_index(i, t32.shape) for i in hottest]
    surface = float(np.mean([t32[c] for c in cells]))
    sx, sy = w0 / GRID[1], h0 / GRID[0]
    result = classify_temperature(surface + config.THERMAL_BODY_OFFSET_C)
    result.update({
        "surface_max_c": round(surface, 1),
        "body_offset_c": config.THERMAL_BODY_OFFSET_C,
        "scale": {"min": tmin, "max": tmax},
        "hotspots": [{"x": round(float((c + 0.5) * sx), 1), "y": round(float((r + 0.5) * sy), 1)} for r, c in cells],
    })
    return result
