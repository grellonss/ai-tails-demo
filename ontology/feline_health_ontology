"""
=============================================================================
Ontologia Biomedica Feline Grimace Scale (FGS) per Explainable AI (XAI)
Libreria: Owlready2 con Reasoner Pellet (supporto SWRL)
IRI Ontologia: http://test.org/feline_health.owl
=============================================================================
Requisiti di sistema:
  - pip install owlready2
  - Java Runtime Environment (JRE) installato e disponibile nel PATH (per Pellet)
=============================================================================
"""

from owlready2 import (
    get_ontology,
    Thing,
    ObjectProperty,
    DataProperty,
    Imp,
    sync_reasoner_pellet
)

# 1. Creazione e inizializzazione dell'ontologia
onto = get_ontology("http://test.org/feline_health.owl")

with onto:
    # -------------------------------------------------------------------------
    # 2. Definizione delle Classi Principali
    # -------------------------------------------------------------------------
    
    # Soggetto principale dell'analisi clinica
    class Cat(Thing):
        """Rappresenta il paziente felino analizzato."""
        pass

    # Classe base per le unità di azione facciale (Facial Features FGS)
    class FacialFeature(Thing):
        """Classe astratta/base per le caratteristiche morfologiche facciali."""
        pass

    # Sottoclassi specifiche per i landmark e action units della FGS
    class EarPosition(FacialFeature):
        """Posizione delle orecchie (dritte, leggermente appiattite o ruotate/abbassate)."""
        pass

    class OrbitalTightening(FacialFeature):
        """Chiusura/tensione orbitale (occhi aperti, semi-chiusi o serrati)."""
        pass

    class MuzzleTension(FacialFeature):
        """Tensione del muso e posizione dei baffi (rilassato, moderatamente teso o ellittico/teso)."""
        pass

    # Punteggio assegnato a ciascuna Action Unit (0, 1, 2)
    class ActionUnitScore(Thing):
        """Livello di intensità osservato per una singola unità d'azione facciale."""
        pass

    # Stato clinico di salute / livello di dolore inferito
    class HealthStatus(Thing):
        """Classificazione semantica del livello di dolore/benessere del felino."""
        pass

    # -------------------------------------------------------------------------
    # 3. Definizione delle Proprietà (Object & Data Properties)
    # -------------------------------------------------------------------------

    # Relazione tra il gatto e le sue caratteristiche facciali
    class hasFeature(ObjectProperty):
        domain = [Cat]
        range = [FacialFeature]

    # Assegnazione del punteggio qualitativo alla singola feature
    class hasScore(ObjectProperty):
        domain = [FacialFeature]
        range = [ActionUnitScore]

    # Stato di salute complessivo assegnato o inferito al gatto
    class hasHealthStatus(ObjectProperty):
        domain = [Cat]
        range = [HealthStatus]

    # Punteggio FGS totale numerico calcolato
    class hasTotalScore(DataProperty):
        domain = [Cat]
        range = [int]

    # -------------------------------------------------------------------------
    # 4. Individui Predefiniti (Istanze)
    # -------------------------------------------------------------------------

    # Individui per ActionUnitScore
    Score_0 = ActionUnitScore("Score_0", comment=["Assente - Landmark nei limiti di normalità (valore 0)"])
    Score_1 = ActionUnitScore("Score_1", comment=["Moderato - Modifica parziale del landmark (valore 1)"])
    Score_2 = ActionUnitScore("Score_2", comment=["Marcato - Chiaro segno di alterazione morfologica (valore 2)"])

    # Individui per HealthStatus
    Good = HealthStatus("Good", comment=["Assenza di dolore (FGS totale <= 1)"])
    Medium = HealthStatus("Medium", comment=["Stato di attenzione / Dolore lieve-moderato (FGS totale 2-3)"])
    Poor = HealthStatus("Poor", comment=["Dolore acuto / Necessità di analgesia (FGS totale >= 4)"])

    # -------------------------------------------------------------------------
    # 5. Regole di Inferenza SWRL (Semantic Web Rule Language)
    # -------------------------------------------------------------------------

    # Regola 1: Totale <= 1 -> Stato 'Good'
    rule_good = Imp()
    rule_good.set_as_rule(
        """Cat(?c) ^ hasTotalScore(?c, ?s) ^ lessThanOrEqual(?s, 1) -> hasHealthStatus(?c, Good)"""
    )

    # Regola 2: Totale >= 2 e <= 3 -> Stato 'Medium'
    rule_medium = Imp()
    rule_medium.set_as_rule(
        """Cat(?c) ^ hasTotalScore(?c, ?s) ^ greaterThanOrEqual(?s, 2) ^ lessThanOrEqual(?s, 3) -> hasHealthStatus(?c, Medium)"""
    )

    # Regola 3: Totale >= 4 -> Stato 'Poor'
    rule_poor = Imp()
    rule_poor.set_as_rule(
        """Cat(?c) ^ hasTotalScore(?c, ?s) ^ greaterThanOrEqual(?s, 4) -> hasHealthStatus(?c, Poor)"""
    )


# -----------------------------------------------------------------------------
# 6. Funzione di Valutazione e Inferenza Semantica
# -----------------------------------------------------------------------------
def evaluate_cat_health(cat_name: str, ear_score_val: int, eye_score_val: int, muzzle_score_val: int, run_reasoner: bool = True):
    """
    Istanzia un paziente felino, popola le sue caratteristiche e invoca il reasoner
    Pellet per dedurre lo stato di salute basato sulle regole SWRL.

    :param cat_name: Nome identificativo dell'individuo gatto
    :param ear_score_val: Punteggio per EarPosition (0, 1 o 2)
    :param eye_score_val: Punteggio per OrbitalTightening (0, 1 o 2)
    :param muzzle_score_val: Punteggio per MuzzleTension (0, 1 o 2)
    :param run_reasoner: Se True, invoca sync_reasoner_pellet()
    :return: Un dizionario esplicativo con l'istanza del gatto e i risultati diagnostici
    """
    score_mapping = {
        0: onto.Score_0,
        1: onto.Score_1,
        2: onto.Score_2
    }

    # Validazione dei parametri di input
    for label, val in [("EarPosition", ear_score_val), ("OrbitalTightening", eye_score_val), ("MuzzleTension", muzzle_score_val)]:
        if val not in score_mapping:
            raise ValueError(f"Valore non valido per {label}: {val}. I valori ammessi per la FGS sono 0, 1 o 2.")

    with onto:
        # A. Istanziazione o recupero del gatto
        cat = onto.Cat(cat_name)

        # B. Istanziazione delle 3 Action Units facciali per il soggetto
        ear_feature = onto.EarPosition(f"{cat_name}_EarPosition")
        eye_feature = onto.OrbitalTightening(f"{cat_name}_OrbitalTightening")
        muzzle_feature = onto.MuzzleTension(f"{cat_name}_MuzzleTension")

        # C. Assegnazione del punteggio (hasScore) a ciascuna feature
        ear_feature.hasScore = [score_mapping[ear_score_val]]
        eye_feature.hasScore = [score_mapping[eye_score_val]]
        muzzle_feature.hasScore = [score_mapping[muzzle_score_val]]

        # D. Collegamento delle feature al soggetto felino (hasFeature)
        cat.hasFeature = [ear_feature, eye_feature, muzzle_feature]

        # E. Calcolo del punteggio totale FGS e assegnazione a hasTotalScore
        total_score = ear_score_val + eye_score_val + muzzle_score_val
        cat.hasTotalScore = [total_score]

    print(f"\n=======================================================")
    print(f" Analisi Clinica FGS: Paziente '{cat_name}'")
    print(f"=======================================================")
    print(f" - Posizione Orecchie (EarPosition):        {ear_score_val} ({score_mapping[ear_score_val].name})")
    print(f" - Tensione Orbitale (OrbitalTightening):  {eye_score_val} ({score_mapping[eye_score_val].name})")
    print(f" - Tensione Muso (MuzzleTension):          {muzzle_score_val} ({score_mapping[muzzle_score_val].name})")
    print(f" - Punteggio Totale FGS:                   {total_score}/6")

    inferred_status_names = []
    clinical_explanation = ""

    if run_reasoner:
        try:
            print(" -> Avvio del reasoner Pellet per l'applicazione delle regole SWRL...")
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
            inferred_status_names = [status.name for status in cat.hasHealthStatus]
            print(f" -> Stato di Salute INFERITO: {inferred_status_names}")
            for status in cat.hasHealthStatus:
                if status.comment:
                    clinical_explanation = status.comment[0]
                    print(f"    Spiegazione: {clinical_explanation}")
        except Exception as e:
            print(f" [!] Avviso Reasoner: Impossibile eseguire Pellet ({e}).")
            # Fallback euristico se Pellet / Java non è disponibile nell'ambiente locale
            if total_score <= 1:
                cat.hasHealthStatus = [onto.Good]
            elif total_score <= 3:
                cat.hasHealthStatus = [onto.Medium]
            else:
                cat.hasHealthStatus = [onto.Poor]
            inferred_status_names = [status.name for status in cat.hasHealthStatus]
            clinical_explanation = cat.hasHealthStatus[0].comment[0]
            print(f" -> Stato applicato da logica di fallback: {inferred_status_names}")

    return {
        "cat_instance": cat,
        "cat_name": cat_name,
        "scores": {
            "ear": ear_score_val,
            "eye": eye_score_val,
            "muzzle": muzzle_score_val,
            "total": total_score
        },
        "inferred_status": inferred_status_names,
        "explanation": clinical_explanation
    }
