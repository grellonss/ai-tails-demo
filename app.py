"""
=============================================================================
AI-Tails: multimodal neuro-symbolic platform (MVP)
=============================================================================
/api/analyze pipeline:
  1. RGB photo     -> YOLO (cat face) -> t4a API (48 landmarks) -> FGS scores (ears, eyes, muzzle)
  2. Thermal photo -> vision/thermal_extractor.py -> estimated temperature + thermal status
  3. OWL2 ontology + SWRL rules (Pellet) -> health status + fired rules
  4. Azure OpenAI -> integrated report
Run: python app.py  (see README for configuration)
=============================================================================
"""

import logging
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import config
from agent import AgentError, chat, generate_multimodal_explanation
from ontology import RULES, evaluate_cat_health, onto
from vision import (FaceModelMissing, LandmarksApiError, ThermalImageError, analyze_landmarks_detailed,
                    analyze_thermal_image, detect_cat_face, send_image_for_processing)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
app = Flask(__name__, template_folder=str(config.PROJECT_ROOT / "templates"),
            static_folder=str(config.PROJECT_ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB per request

# How each ontology health status is shown to the user
VERDICTS = {
    "Good": {"level": "good", "is_sick": False, "label": "NOT SICK (HEALTHY)"},
    "Medium": {"level": "medium", "is_sick": False, "label": "NEEDS ATTENTION"},
    "Poor": {"level": "poor", "is_sick": True, "label": "SICK (ACUTE PAIN)"},
    "AcuteInflammatoryPain": {"level": "poor", "is_sick": True, "label": "SICK (PAIN + FEVER)"},
}


def _error(message: str, status: int, step: str):
    return jsonify({"error": message, "step": step}), status


def _decode(file_storage):
    data = file_storage.read()
    return data, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Component status: useful to check the configuration before the demo."""
    return jsonify({
        "status": "online",
        "ontology": str(onto.base_iri),
        "rules": [r.name for r in RULES],
        "yolo_weights": config.YOLO_WEIGHTS.exists(),
        "landmarks_api": bool(config.LANDMARKS_API_URL),
        "azure_openai": config.azure_configured(),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Multipart form: image (RGB face photo), thermal_image (thermal photo), cat_name (optional).
    Returns FGS, thermal, ontology outcome and report; on error {"error", "step"}.
    """
    cat_name = (request.form.get("cat_name") or "").strip() or "Patient"
    rgb_file, thermal_file = request.files.get("image"), request.files.get("thermal_image")
    if not rgb_file or not thermal_file:
        return _error("Both photos are required: 'image' (RGB) and 'thermal_image' (thermal).", 400, "input")

    # 1. Face: YOLO -> landmarks -> FGS
    rgb_bytes, rgb = _decode(rgb_file)
    if rgb is None:
        return _error("The RGB photo could not be read.", 400, "input")
    try:
        face = detect_cat_face(rgb)
    except FaceModelMissing as e:
        return _error(str(e), 500, "face")
    if face is None:
        return _error("No cat face found in the photo: a close-up, frontal photo is needed.", 422, "face")
    meta = {"source": "api"}
    try:
        landmarks = send_image_for_processing(rgb_bytes, config.LANDMARKS_API_URL,
                                              name=rgb_file.filename or "image.jpg", face_box=face["box"], meta=meta)
    except LandmarksApiError as e:
        return _error(str(e), 502, "landmarks")
    fgs = analyze_landmarks_detailed(landmarks)

    # 2. Thermal photo -> temperature
    _, thermal_img = _decode(thermal_file)
    try:
        thermal = analyze_thermal_image(thermal_img)
    except ThermalImageError as e:
        return _error(str(e), 422, "thermal")

    # 3. Ontology: the status and the fired rules come from the reasoner, not from this file
    ontology = evaluate_cat_health(cat_name, fgs["scores"]["ear"], fgs["scores"]["eye"], fgs["scores"]["muzzle"],
                                   temperature=thermal["temperature"], thermal_status=thermal["thermal_status"])

    # 4. Report
    report = generate_multimodal_explanation(cat_name, fgs["scores"], thermal, ontology)

    return jsonify({
        "cat_name": cat_name,
        "verdict": VERDICTS[ontology["primary_status"]],
        "face": face,
        "landmarks": landmarks,
        "landmarks_source": meta["source"],
        "scores": fgs["scores"], "ratios": fgs["ratios"], "descriptions": fgs["descriptions"],
        "thermal": thermal,
        "ontology": ontology,
        "report": report,
    })


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """JSON {question, context (/api/analyze response), history [{role, content}]} -> {answer, engine}."""
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return _error("Empty question.", 400, "chat")
    context = body.get("context") or {}
    context = {k: context.get(k) for k in ("cat_name", "verdict", "scores", "thermal", "ontology", "report")}
    try:
        return jsonify(chat(question, context, body.get("history")))
    except AgentError as e:
        return _error(str(e), 502, "chat")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n AI-Tails listening on http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
