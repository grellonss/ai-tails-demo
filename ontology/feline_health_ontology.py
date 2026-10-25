"""
=============================================================================
Biomedical Feline Grimace Scale (FGS) ontology + thermal extension, for Explainable AI
Library: Owlready2 with the Pellet reasoner (SWRL support)
Ontology IRI: http://test.org/feline_health.owl
=============================================================================
Requirements: pip install owlready2  +  Java (JRE) on PATH for Pellet.
The SWRL rules are in ontology/rules.py.
=============================================================================
"""

import logging
import re
import threading
import uuid
from typing import Optional

from owlready2 import (DataProperty, Imp, ObjectProperty, Thing, destroy_entity, get_ontology,
                       sync_reasoner_pellet)

from .rules import RULES, SEVERITY

log = logging.getLogger(__name__)
onto = get_ontology("http://test.org/feline_health.owl")

with onto:
    # -------------------------------------------------------------------------
    # Classes
    # -------------------------------------------------------------------------
    class Cat(Thing):
        """The feline patient being analysed."""

    class FacialFeature(Thing):
        """Base class for the FGS facial action units."""

    class EarPosition(FacialFeature):
        """Ear position (upright, slightly flattened or rotated/lowered)."""

    class OrbitalTightening(FacialFeature):
        """Orbital tightening (eyes open, half-closed or squeezed)."""

    class MuzzleTension(FacialFeature):
        """Muzzle tension (relaxed, moderately tense or elliptical/tense)."""

    class ActionUnitScore(Thing):
        """Observed intensity of a single action unit (0, 1, 2)."""

    class HealthStatus(Thing):
        """Inferred health status / pain level."""

    class ThermalStatus(Thing):
        """Thermal status produced by the thermal module (vision/thermal_extractor.py)."""

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    class hasFeature(ObjectProperty):
        domain = [Cat]
        range = [FacialFeature]

    class hasScore(ObjectProperty):
        domain = [FacialFeature]
        range = [ActionUnitScore]

    class hasHealthStatus(ObjectProperty):
        domain = [Cat]
        range = [HealthStatus]

    class hasTotalScore(DataProperty):
        """Total FGS score (0-6)."""
        domain = [Cat]
        range = [int]

    class hasTemperature(DataProperty):
        """Body temperature estimated from the thermal photo (°C)."""
        domain = [Cat]
        range = [float]

    class hasThermalStatus(ObjectProperty):
        domain = [Cat]
        range = [ThermalStatus]

    # -------------------------------------------------------------------------
    # Individuals
    # -------------------------------------------------------------------------
    Score_0 = ActionUnitScore("Score_0", comment=["Absent - Landmark within normal limits (value 0)"])
    Score_1 = ActionUnitScore("Score_1", comment=["Moderate - Partial change of the landmark (value 1)"])
    Score_2 = ActionUnitScore("Score_2", comment=["Marked - Clear sign of morphological change (value 2)"])

    Good = HealthStatus("Good", comment=["No pain (total FGS <= 1)"])
    Medium = HealthStatus("Medium", comment=["Needs attention / Mild-to-moderate pain (total FGS 2-3)"])
    Poor = HealthStatus("Poor", comment=["Acute pain / Analgesia needed (total FGS >= 4)"])
    AcuteInflammatoryPain = HealthStatus("AcuteInflammatoryPain", comment=[
        "Acute pain with fever (FGS >= 4 and temperature >= 39.5 °C): active inflammatory process"])

    Hypothermia = ThermalStatus("Hypothermia", comment=["Temperature < 38.0 °C"])
    Normothermia = ThermalStatus("Normothermia", comment=["Temperature 38.0 - 39.2 °C"])
    MildFever = ThermalStatus("MildFever", comment=["Mild fever 39.3 - 39.9 °C"])
    HighFever = ThermalStatus("HighFever", comment=["High fever >= 40.0 °C"])

    # -------------------------------------------------------------------------
    # SWRL rules (text in ontology/rules.py)
    # -------------------------------------------------------------------------
    for _rule in RULES:
        Imp(name=f"rule_{_rule.name}").set_as_rule(_rule.swrl)

SCORES = {0: Score_0, 1: Score_1, 2: Score_2}
FEATURES = {"ear": EarPosition, "eye": OrbitalTightening, "muzzle": MuzzleTension}
_lock = threading.Lock()  # the Owlready world is global: one inference at a time


def _rules_for(statuses: list[str], total: int, temperature: Optional[float]) -> list[dict]:
    return [{"name": r.name, "status": r.status, "swrl": r.swrl}
            for r in RULES if r.status in statuses and r.holds(total, temperature)]


def evaluate_cat_health(cat_name: str, ear_score_val: int, eye_score_val: int, muzzle_score_val: int,
                        temperature: Optional[float] = None, thermal_status: Optional[str] = None,
                        run_reasoner: bool = True) -> dict:
    """
    Instantiates the cat with its action units (and, if available, the temperature), runs Pellet
    to apply the SWRL rules and returns the inferred health status with the fired rules.
    The individuals created are removed at the end, so analyses do not affect each other.

    :return: {cat_name, scores, temperature, thermal_status, inferred_status (least to most severe),
              primary_status, explanation, fired_rules [{name, status, swrl}], reasoner}
    """
    values = {"ear": ear_score_val, "eye": eye_score_val, "muzzle": muzzle_score_val}
    for label, val in values.items():
        if val not in SCORES:
            raise ValueError(f"Invalid value for {label}: {val}. Allowed FGS values are 0, 1 or 2.")
    total = sum(values.values())
    if thermal_status is not None and onto[thermal_status] is None:
        raise ValueError(f"Unknown thermal status: {thermal_status}")

    with _lock:
        base = re.sub(r"\W+", "_", cat_name).strip("_") or "Cat"
        uid = uuid.uuid4().hex[:8]
        with onto:
            cat = Cat(f"{base}_{uid}")
            features = []
            for key, cls in FEATURES.items():
                feature = cls(f"{base}_{uid}_{cls.__name__}")
                feature.hasScore = [SCORES[values[key]]]
                features.append(feature)
            cat.hasFeature = features
            cat.hasTotalScore = [total]
            if temperature is not None:
                cat.hasTemperature = [float(temperature)]
            if thermal_status is not None:
                cat.hasThermalStatus = [onto[thermal_status]]

        reasoner = "python-fallback"
        statuses: list[str] = []
        if run_reasoner:
            try:
                sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=0)
                statuses = [s.name for s in cat.hasHealthStatus]
                reasoner = "pellet"
            except Exception as e:  # Java/Pellet not available: same rules, evaluated in Python
                log.warning("Pellet not available (%s): using the Python fallback of the same rules.", e)
        if reasoner == "python-fallback":
            statuses = [r.status for r in RULES if r.holds(total, temperature)]

        for entity in [cat, *features]:
            destroy_entity(entity)

    statuses = sorted(set(statuses), key=SEVERITY.index)
    primary = statuses[-1] if statuses else "Good"
    return {
        "cat_name": cat_name,
        "scores": {**values, "total": total},
        "temperature": temperature,
        "thermal_status": thermal_status,
        "inferred_status": statuses,
        "primary_status": primary,
        "explanation": onto[primary].comment[0],
        "fired_rules": _rules_for(statuses, total, temperature),
        "reasoner": reasoner,
    }
