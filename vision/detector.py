"""
Modulo Vision per AI-Tails.
Ponte Geometrico tra l'API dei 48 Landmark Facciali e l'Ontologia FGS.
"""

import math
import base64
from typing import Dict, Any, Tuple, List, Union


# =============================================================================
# 1. FUNZIONI DI UTILITÀ GEOMETRICA
# =============================================================================

def calc_distance(p1: Union[Dict[str, float], List[float], Tuple[float, float]], 
                  p2: Union[Dict[str, float], List[float], Tuple[float, float]]) -> float:
    """
    Calcola la distanza euclidea tra due punti bidimensionali (x, y).
    Supporta sia dizionari {'x': ..., 'y': ...} sia tuple/liste (x, y).
    """
    if isinstance(p1, dict) and 'x' in p1 and 'y' in p1:
        x1, y1 = p1['x'], p1['y']
    elif isinstance(p1, (list, tuple)) and len(p1) >= 2:
        x1, y1 = p1[0], p1[1]
    else:
        raise ValueError(f"Formato punto p1 non valido: {p1}")

    if isinstance(p2, dict) and 'x' in p2 and 'y' in p2:
        x2, y2 = p2['x'], p2['y']
    elif isinstance(p2, (list, tuple)) and len(p2) >= 2:
        x2, y2 = p2[0], p2[1]
    else:
        raise ValueError(f"Formato punto p2 non valido: {p2}")

    return math.dist((x1, y1), (x2, y2))


# =============================================================================
# 2. MAPPATURA INDICI LANDMARK (Calibrabili per Hackathon)
# =============================================================================
# NOTA PER IL TEAM (Aldo & Fabio):
# Gli indici 0-47 corrispondono ai 48 punti dell'API t4a.
# Se durante i test sull'immagine verificate indici diversi, potete aggiornare
# direttamente questo dizionario senza modificare la logica di calcolo.
DEFAULT_LANDMARK_MAP = {
    # Occhio (Eye Aspect Ratio - EAR): altezza vs larghezza
    "eye_top": 10,       # Palpebra superiore
    "eye_bottom": 14,    # Palpebra inferiore
    "eye_inner": 8,      # Angolo interno occhio
    "eye_outer": 12,     # Angolo esterno occhio

    # Orecchie: distanza punte orecchie rispetto alla larghezza cranica
    "ear_tip_left": 2,   # Punta orecchio sinistro
    "ear_tip_right": 40, # Punta orecchio destro
    "head_left": 0,      # Lato sinistro testa
    "head_right": 25,    # Lato destro testa

    # Muso: rapporto altezza / larghezza (rotondo vs ellittico/teso)
    "muzzle_top": 30,    # Base naso / parte superiore muso
    "muzzle_bottom": 35, # Margine inferiore mento/muso
    "muzzle_left": 32,   # Cuscinetto baffi sinistro
    "muzzle_right": 37   # Cuscinetto baffi destro
}


# =============================================================================
# 3. IL PONTE GEOMETRICO (Feature Engineering -> Punteggi FGS)
# =============================================================================

def extract_fgs_scores_from_landmarks(landmarks: Union[List[Dict[str, float]], Dict[int, Dict[str, float]]],
                                      indices_map: Dict[str, int] = None) -> Tuple[int, int, int]:
    """
    Riceve i 48 landmark dall'API ed esegue il Feature Engineering geometrico,
    restituendo i punteggi discreti (0, 1, 2) conformi alla Feline Grimace Scale.

    :param landmarks: Lista o dizionario di punti facciali [{'x':..., 'y':...}, ...]
    :param indices_map: (Opzionale) Mappatura personalizzata degli indici landmark
    :return: Tupla con (ear_score, eye_score, muzzle_score)
    """
    mapping = indices_map if indices_map is not None else DEFAULT_LANDMARK_MAP

    # -------------------------------------------------------------------------
    # 1. ORBITAL TIGHTENING (Chiusura Occhi)
    # Calcolo dell'Eye Aspect Ratio (Altezza / Larghezza)
    # -------------------------------------------------------------------------
    eye_height = calc_distance(landmarks[mapping["eye_top"]], landmarks[mapping["eye_bottom"]])
    eye_width = calc_distance(landmarks[mapping["eye_inner"]], landmarks[mapping["eye_outer"]])

    eye_ratio = (eye_height / eye_width) if eye_width > 0 else 0.0

    if eye_ratio > 0.6:
        eye_score = 0  # Occhio ben aperto (Assente)
    elif 0.4 <= eye_ratio <= 0.6:
        eye_score = 1  # Occhio socchiuso (Moderato)
    else:
        eye_score = 2  # Occhio a fessura (Marcato)

    # -------------------------------------------------------------------------
    # 2. EAR POSITION (Posizione Orecchie)
    # Distanza tra le punte delle orecchie normalizzata sulla larghezza della testa
    # -------------------------------------------------------------------------
    ear_tips_dist = calc_distance(landmarks[mapping["ear_tip_left"]], landmarks[mapping["ear_tip_right"]])
    head_width = calc_distance(landmarks[mapping["head_left"]], landmarks[mapping["head_right"]])

    ear_ratio = (ear_tips_dist / head_width) if head_width > 0 else 0.0

    if ear_ratio < 1.2:
        ear_score = 0  # Orecchie dritte (Assente)
    elif 1.2 <= ear_ratio <= 1.5:
        ear_score = 1  # Orecchie leggermente allargate (Moderato)
    else:
        ear_score = 2  # Orecchie piatte / ruotate all'infuori (Marcato)

    # -------------------------------------------------------------------------
    # 3. MUZZLE TENSION (Tensione del Muso)
    # Forma del muso (rotonda vs ellittica/orizzontale)
    # -------------------------------------------------------------------------
    muzzle_height = calc_distance(landmarks[mapping["muzzle_top"]], landmarks[mapping["muzzle_bottom"]])
    muzzle_width = calc_distance(landmarks[mapping["muzzle_left"]], landmarks[mapping["muzzle_right"]])

    muzzle_ratio = (muzzle_height / muzzle_width) if muzzle_width > 0 else 0.0

    if muzzle_ratio > 0.8:
        muzzle_score = 0  # Muso rilassato / Rotondo (Assente)
    elif 0.6 <= muzzle_ratio <= 0.8:
        muzzle_score = 1  # Lieve appiattimento (Moderato)
    else:
        muzzle_score = 2  # Teso / Ellittico (Marcato)

    return ear_score, eye_score, muzzle_score


def analyze_landmarks_detailed(landmarks: Union[List[Dict[str, float]], Dict[int, Dict[str, float]]],
                               indices_map: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Esegue l'analisi completa restituendo sia gli score discreti sia i valori biometrici continui.
    """
    mapping = indices_map if indices_map is not None else DEFAULT_LANDMARK_MAP

    # 1. ORBITAL TIGHTENING (Chiusura Occhi)
    eye_height = calc_distance(landmarks[mapping["eye_top"]], landmarks[mapping["eye_bottom"]])
    eye_width = calc_distance(landmarks[mapping["eye_inner"]], landmarks[mapping["eye_outer"]])
    eye_ratio = round((eye_height / eye_width), 3) if eye_width > 0 else 0.0

    if eye_ratio > 0.6:
        eye_score = 0
        eye_desc = "Occhio ben aperto (Assente)"
    elif 0.4 <= eye_ratio <= 0.6:
        eye_score = 1
        eye_desc = "Occhio socchiuso (Moderato)"
    else:
        eye_score = 2
        eye_desc = "Occhio a fessura (Marcato)"

    # 2. EAR POSITION (Posizione Orecchie)
    ear_tips_dist = calc_distance(landmarks[mapping["ear_tip_left"]], landmarks[mapping["ear_tip_right"]])
    head_width = calc_distance(landmarks[mapping["head_left"]], landmarks[mapping["head_right"]])
    ear_ratio = round((ear_tips_dist / head_width), 3) if head_width > 0 else 0.0

    if ear_ratio < 1.2:
        ear_score = 0
        ear_desc = "Orecchie dritte (Assente)"
    elif 1.2 <= ear_ratio <= 1.5:
        ear_score = 1
        ear_desc = "Orecchie leggermente allargate (Moderato)"
    else:
        ear_score = 2
        ear_desc = "Orecchie piatte / ruotate (Marcato)"

    # 3. MUZZLE TENSION (Tensione del Muso)
    muzzle_height = calc_distance(landmarks[mapping["muzzle_top"]], landmarks[mapping["muzzle_bottom"]])
    muzzle_width = calc_distance(landmarks[mapping["muzzle_left"]], landmarks[mapping["muzzle_right"]])
    muzzle_ratio = round((muzzle_height / muzzle_width), 3) if muzzle_width > 0 else 0.0

    if muzzle_ratio > 0.8:
        muzzle_score = 0
        muzzle_desc = "Rilassato / Rotondo (Assente)"
    elif 0.6 <= muzzle_ratio <= 0.8:
        muzzle_score = 1
        muzzle_desc = "Lieve tensione (Moderato)"
    else:
        muzzle_score = 2
        muzzle_desc = "Teso / Ellittico (Marcato)"

    return {
        "scores": {
            "ear": ear_score,
            "eye": eye_score,
            "muzzle": muzzle_score,
            "total": ear_score + eye_score + muzzle_score
        },
        "ratios": {
            "ear_ratio": ear_ratio,
            "eye_ratio": eye_ratio,
            "muzzle_ratio": muzzle_ratio
        },
        "descriptions": {
            "ear": ear_desc,
            "eye": eye_desc,
            "muzzle": muzzle_desc
        }
    }


def generate_landmarks_for_image(width: float, height: float, profile: str = "healthy") -> List[Dict[str, float]]:
    """
    Genera 48 coordinate proporzionate alla larghezza e altezza dell'immagine caricata.
    """
    cx = width * 0.5
    cy = height * 0.52
    scale = min(width, height) * 0.45

    landmarks = [{"x": cx, "y": cy} for _ in range(48)]

    # Larghezza testa proporzionata
    landmarks[DEFAULT_LANDMARK_MAP["head_left"]] = {"x": cx - scale * 0.6, "y": cy - scale * 0.1}
    landmarks[DEFAULT_LANDMARK_MAP["head_right"]] = {"x": cx + scale * 0.6, "y": cy - scale * 0.1}

    # Occhio
    landmarks[DEFAULT_LANDMARK_MAP["eye_inner"]] = {"x": cx - scale * 0.28, "y": cy - scale * 0.12}
    landmarks[DEFAULT_LANDMARK_MAP["eye_outer"]] = {"x": cx - scale * 0.08, "y": cy - scale * 0.12}

    # Muso
    landmarks[DEFAULT_LANDMARK_MAP["muzzle_left"]] = {"x": cx - scale * 0.22, "y": cy + scale * 0.25}
    landmarks[DEFAULT_LANDMARK_MAP["muzzle_right"]] = {"x": cx + scale * 0.22, "y": cy + scale * 0.25}

    if profile == "healthy":
        landmarks[DEFAULT_LANDMARK_MAP["eye_top"]] = {"x": cx - scale * 0.18, "y": cy - scale * 0.20}
        landmarks[DEFAULT_LANDMARK_MAP["eye_bottom"]] = {"x": cx - scale * 0.18, "y": cy - scale * 0.04}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_left"]] = {"x": cx - scale * 0.52, "y": cy - scale * 0.72}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_right"]] = {"x": cx + scale * 0.52, "y": cy - scale * 0.72}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_top"]] = {"x": cx, "y": cy + scale * 0.08}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_bottom"]] = {"x": cx, "y": cy + scale * 0.44}

    elif profile == "moderate":
        landmarks[DEFAULT_LANDMARK_MAP["eye_top"]] = {"x": cx - scale * 0.18, "y": cy - scale * 0.17}
        landmarks[DEFAULT_LANDMARK_MAP["eye_bottom"]] = {"x": cx - scale * 0.18, "y": cy - scale * 0.07}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_left"]] = {"x": cx - scale * 0.72, "y": cy - scale * 0.60}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_right"]] = {"x": cx + scale * 0.72, "y": cy - scale * 0.60}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_top"]] = {"x": cx, "y": cy + scale * 0.12}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_bottom"]] = {"x": cx, "y": cy + scale * 0.41}

    else:  # pain / marcato
        landmarks[DEFAULT_LANDMARK_MAP["eye_top"]] = {"x": cx - scale * 0.18, "y": cy - scale * 0.14}
        landmarks[DEFAULT_LANDMARK_MAP["eye_bottom"]] = {"x": cx - scale * 0.18, "y": cy - scale * 0.10}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_left"]] = {"x": cx - scale * 0.95, "y": cy - scale * 0.45}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_right"]] = {"x": cx + scale * 0.95, "y": cy - scale * 0.45}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_top"]] = {"x": cx, "y": cy + scale * 0.16}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_bottom"]] = {"x": cx, "y": cy + scale * 0.34}

    return landmarks


# =============================================================================
# 4. CLIENT API T4A LANDMARKS (Invio Foto -> Estrazione Punti)
# =============================================================================

def send_image_for_processing(image_path: str, api_url: str, api_key: str = None) -> List[Dict[str, float]]:
    """
    Carica l'immagine, la converte in Base64 e la invia all'API t4a Animal Facial Landmarks.
    Estrae e restituisce la lista di 48 landmark del gatto.

    :param image_path: Percorso del file immagine locale
    :param api_url: Endpoint URL dell'API di rilevamento
    :param api_key: (Opzionale) Chiave API per autorizzazione
    :return: Lista di 48 coordinate [{'x': ..., 'y': ...}, ...]
    """
    import requests

    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "image": encoded_image
    }

    response = requests.post(api_url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    # Estrazione dei landmark secondo il formato API
    # Supporta sia formati come response_json[0]['cat']['landmarks'] sia risposte piatte
    if isinstance(data, list) and len(data) > 0 and 'cat' in data[0]:
        return data[0]['cat']['landmarks']
    elif isinstance(data, dict) and 'landmarks' in data:
        return data['landmarks']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Formato risposta API inatteso: {data.keys() if isinstance(data, dict) else type(data)}")


# =============================================================================
# 5. MOCK DATA GENERATOR (Per test locali ed esecuzione offline)
# =============================================================================

def generate_mock_landmarks(profile: str = "healthy") -> List[Dict[str, float]]:
    """
    Genera 48 landmark sintetici realistici per testare la pipeline
    anche in assenza di connessione all'API remota durante l'hackathon.

    :param profile: 'healthy' (Score 0-1), 'moderate' (Score 2-3), 'pain' (Score >= 4)
    """
    landmarks = [{"x": 100.0, "y": 100.0} for _ in range(48)]

    # Larghezza testa base: da x=50 a x=250 (distanza = 200)
    landmarks[DEFAULT_LANDMARK_MAP["head_left"]] = {"x": 50.0, "y": 150.0}
    landmarks[DEFAULT_LANDMARK_MAP["head_right"]] = {"x": 250.0, "y": 150.0}

    # Larghezza occhio fissa: 30 px
    landmarks[DEFAULT_LANDMARK_MAP["eye_inner"]] = {"x": 100.0, "y": 130.0}
    landmarks[DEFAULT_LANDMARK_MAP["eye_outer"]] = {"x": 130.0, "y": 130.0}

    # Larghezza muso fissa: 50 px
    landmarks[DEFAULT_LANDMARK_MAP["muzzle_left"]] = {"x": 125.0, "y": 200.0}
    landmarks[DEFAULT_LANDMARK_MAP["muzzle_right"]] = {"x": 175.0, "y": 200.0}

    if profile == "healthy":
        # Occhi aperti: altezza 22 (ratio ~ 0.73 > 0.6 -> score 0)
        landmarks[DEFAULT_LANDMARK_MAP["eye_top"]] = {"x": 115.0, "y": 119.0}
        landmarks[DEFAULT_LANDMARK_MAP["eye_bottom"]] = {"x": 115.0, "y": 141.0}
        # Orecchie dritte: distanza 200 (ratio 200/200 = 1.0 < 1.2 -> score 0)
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_left"]] = {"x": 50.0, "y": 50.0}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_right"]] = {"x": 250.0, "y": 50.0}
        # Muso tondo: altezza 45 (ratio 45/50 = 0.9 > 0.8 -> score 0)
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_top"]] = {"x": 150.0, "y": 177.0}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_bottom"]] = {"x": 150.0, "y": 222.0}

    elif profile == "moderate":
        # Occhi socchiusi: altezza 15 (ratio 15/30 = 0.5 -> score 1)
        landmarks[DEFAULT_LANDMARK_MAP["eye_top"]] = {"x": 115.0, "y": 122.5}
        landmarks[DEFAULT_LANDMARK_MAP["eye_bottom"]] = {"x": 115.0, "y": 137.5}
        # Orecchie allargate: distanza 260 (ratio 260/200 = 1.3 -> score 1)
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_left"]] = {"x": 20.0, "y": 70.0}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_right"]] = {"x": 280.0, "y": 70.0}
        # Muso lieve tensione: altezza 35 (ratio 35/50 = 0.7 -> score 1)
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_top"]] = {"x": 150.0, "y": 182.5}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_bottom"]] = {"x": 150.0, "y": 217.5}

    else:  # pain (marcato)
        # Occhi a fessura: altezza 9 (ratio 9/30 = 0.3 < 0.4 -> score 2)
        landmarks[DEFAULT_LANDMARK_MAP["eye_top"]] = {"x": 115.0, "y": 125.5}
        landmarks[DEFAULT_LANDMARK_MAP["eye_bottom"]] = {"x": 115.0, "y": 134.5}
        # Orecchie piatte: distanza 320 (ratio 320/200 = 1.6 > 1.5 -> score 2)
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_left"]] = {"x": 0.0, "y": 100.0}
        landmarks[DEFAULT_LANDMARK_MAP["ear_tip_right"]] = {"x": 320.0, "y": 100.0}
        # Muso molto ellittico: altezza 25 (ratio 25/50 = 0.5 < 0.6 -> score 2)
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_top"]] = {"x": 150.0, "y": 187.5}
        landmarks[DEFAULT_LANDMARK_MAP["muzzle_bottom"]] = {"x": 150.0, "y": 212.5}

    return landmarks
