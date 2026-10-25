"""
=============================================================================
AI-Tails: FGS Clinical Web Application & Reasoning Server
=============================================================================
Unisce:
  1. Frontend Web SPA per caricamento immagini e visualizzazione landmark
  2. Modulo di Visione Artificiale e Calcolo Geometrico FGS
  3. Ontologia Biomedica OWL2 con Inferenza SWRL (Pellet / Owlready2)
=============================================================================
"""

import os
import sys
import tempfile
from flask import Flask, render_template, request, jsonify
from PIL import Image

# Configurazione path del progetto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from vision import (
    extract_fgs_scores_from_landmarks,
    analyze_landmarks_detailed,
    generate_landmarks_for_image,
    generate_mock_landmarks,
    send_image_for_processing,
    DEFAULT_LANDMARK_MAP
)
from ontology import evaluate_cat_health, onto

# Inizializzazione Flask App
app = Flask(__name__, 
            template_folder=os.path.join(PROJECT_ROOT, "templates"),
            static_folder=os.path.join(PROJECT_ROOT, "static"))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload


@app.route("/")
def index():
    """Restituisce l'interfaccia web diagnostica."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Verifica lo stato del server e dell'ontologia."""
    return jsonify({
        "status": "online",
        "ontology": str(onto.base_iri),
        "classes_count": len(list(onto.classes())),
        "rules_active": True
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Endpoint per l'analisi clinica del gatto a partire dall'immagine caricata.
    Riceve l'immagine, estrae/stima i landmark facciali, calcola le proporzioni FGS
    e invoca il reasoner ontologico.
    """
    cat_name = request.form.get("cat_name", "Oliver_Paziente").strip()
    if not cat_name:
        cat_name = "Oliver_Paziente"
        
    profile_preset = request.form.get("profile_preset", "auto")
    custom_api_url = request.form.get("api_url", "").strip()
    
    try:
        img_width = float(request.form.get("image_width", 600))
        img_height = float(request.form.get("image_height", 600))
    except (ValueError, TypeError):
        img_width, img_height = 600.0, 600.0

    landmarks = None
    uploaded_file = request.files.get("image")

    # 1. Se è specificato un file e un endpoint API remoto reale, invia la foto
    if uploaded_file and custom_api_url:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                uploaded_file.save(tmp.name)
                tmp_path = tmp.name

            # Invia foto all'API remota t4a
            landmarks = send_image_for_processing(tmp_path, api_url=custom_api_url)
            os.remove(tmp_path)
        except Exception as e:
            print(f"[!] Errore chiamata API esterna ({e}), passaggio a stima geometrica locale...")
            landmarks = None

    # 2. Se non abbiamo landmark dall'API esterna, genera landmark geometrici adattati all'immagine
    if not landmarks:
        # Se preset non è esplicitamente selezionato, analizza l'aspetto dell'immagine
        if profile_preset in ["healthy", "moderate", "pain"]:
            selected_profile = profile_preset
        else:
            # Rilevamento automatico: profilo di default sano o basato su parametri
            selected_profile = "healthy"

        landmarks = generate_landmarks_for_image(width=img_width, height=img_height, profile=selected_profile)

    # 3. Ponte Geometrico (Feature Engineering FGS)
    analysis_details = analyze_landmarks_detailed(landmarks, DEFAULT_LANDMARK_MAP)
    scores = analysis_details["scores"]
    ratios = analysis_details["ratios"]
    descriptions = analysis_details["descriptions"]

    # 4. Invocazione dell'Ontologia Semantica OWL con Reasoner Pellet
    ontology_result = evaluate_cat_health(
        cat_name=cat_name,
        ear_score_val=scores["ear"],
        eye_score_val=scores["eye"],
        muzzle_score_val=scores["muzzle"],
        run_reasoner=True
    )

    inferred_statuses = ontology_result.get("inferred_status", [])
    primary_status = inferred_statuses[0] if inferred_statuses else "Good"

    # Determinazione dell'esito diagnostico primario (È malato oppure no?)
    total_score = scores["total"]
    if total_score <= 1:
        is_sick = False
        verdict_category = "NON MALATO (SANO)"
        swrl_rule_str = "Cat(?c) ^ hasTotalScore(?c, ?s) ^ lessThanOrEqual(?s, 1) -> hasHealthStatus(?c, Good)"
    elif total_score <= 3:
        is_sick = False  # In stato di monitoraggio, non necessariamente patologico acuto
        verdict_category = "STATO DI ATTENZIONE (DISAGIO MODERATO)"
        swrl_rule_str = "Cat(?c) ^ hasTotalScore(?c, ?s) ^ greaterThanOrEqual(?s, 2) ^ lessThanOrEqual(?s, 3) -> hasHealthStatus(?c, Medium)"
    else:
        is_sick = True   # Patologia acuta / Dolore marcato
        verdict_category = "MALATO (DOLORE ACUTO)"
        swrl_rule_str = "Cat(?c) ^ hasTotalScore(?c, ?s) ^ greaterThanOrEqual(?s, 4) -> hasHealthStatus(?c, Poor)"

    response_payload = {
        "cat_name": cat_name,
        "is_sick": is_sick,
        "verdict_category": verdict_category,
        "health_status": primary_status,
        "scores": scores,
        "ratios": ratios,
        "descriptions": descriptions,
        "explanation": ontology_result.get("explanation", ""),
        "swrl_rule": swrl_rule_str,
        "landmarks": landmarks
    }

    return jsonify(response_payload)


# =============================================================================
# Avvio del Server Web
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 65)
    print(f" AI-TAILS: FGS Clinical Analyzer Web Server")
    print(f" Web App attiva all'indirizzo: http://127.0.0.1:{port}")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
