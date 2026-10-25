"""
DEVELOPMENT ONLY: fake t4a landmarks API server, with the same request/response format.
Used to try the pipeline when the real API is unreachable. The points do NOT come from the photo:
they are synthetic landmarks with known FGS scores, placed inside the face found by YOLO.

    python -m tools.mock_landmarks_server --profile pain      # healthy | moderate | pain
    # then in .env:  LANDMARKS_API_URL=http://127.0.0.1:6400/landmarks
"""

import argparse
import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request

from vision import DEFAULT_LANDMARK_MAP as M, detect_cat_face, generate_mock_landmarks

app = Flask(__name__)
PROFILE = "healthy"

# Where each feature sits in a YOLO face box (fractions of width, height), measured on a frontal photo.
# Each group is moved as a whole, so the FGS ratios of the synthetic profile do not change.
ANCHORS = {
    ("ear_tip_left", "ear_tip_right"): (0.50, 0.16),
    ("head_left", "head_right"): (0.50, 0.60),
    ("eye_top", "eye_bottom", "eye_inner", "eye_outer"): (0.39, 0.58),
    ("muzzle_top", "muzzle_bottom", "muzzle_left", "muzzle_right"): (0.50, 0.77),
}
HEAD_WIDTH = 0.74   # cheek-to-cheek width as a fraction of the box width (template head = 200 units)


def _place_in_box(template: list[dict], box: dict) -> list[dict]:
    w, h = box["x2"] - box["x1"], box["y2"] - box["y1"]
    s = HEAD_WIDTH * w / 200
    pts = [{"x": p["x"] * s, "y": p["y"] * s} for p in template]
    for names, (fx, fy) in ANCHORS.items():
        idx = [M[n] for n in names]
        cx, cy = (sum(pts[i][k] for i in idx) / len(idx) for k in ("x", "y"))
        dx, dy = box["x1"] + fx * w - cx, box["y1"] + fy * h - cy
        for i in idx:
            pts[i] = {"x": pts[i]["x"] + dx, "y": pts[i]["y"] + dy}
    used = set(M.values())
    # points not used by the FGS are hidden under the muzzle_top marker instead of floating on the photo
    return [p if i in used else dict(pts[M["muzzle_top"]]) for i, p in enumerate(pts)]


@app.post("/landmarks")
def landmarks():
    data_url = (request.get_json(silent=True) or {}).get("image", "")
    raw = base64.b64decode(data_url.split(",", 1)[-1] or b"")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "invalid image"}), 400
    face = detect_cat_face(img)
    box = face["box"] if face else {"x1": 0, "y1": 0, "x2": img.shape[1], "y2": img.shape[0]}
    resp = jsonify([{"cat": {"landmarks": _place_in_box(generate_mock_landmarks(PROFILE), box)}}])
    resp.headers["X-Mock-Landmarks"] = PROFILE        # the app shows this: the points do not come from the photo
    return resp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="healthy", choices=["healthy", "moderate", "pain"])
    ap.add_argument("--port", type=int, default=6400)
    args = ap.parse_args()
    PROFILE = args.profile
    print(f"MOCK landmarks API (profile {PROFILE}) on http://127.0.0.1:{args.port}/landmarks")
    app.run(host="127.0.0.1", port=args.port)
