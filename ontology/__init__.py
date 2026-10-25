"""
Modulo Ontology per AI-Tails.
Esporta l'ontologia basata sulla Feline Grimace Scale (FGS) e il classificatore semantico.
"""

from .feline_health_ontology import (
    onto,
    Cat,
    FacialFeature,
    EarPosition,
    OrbitalTightening,
    MuzzleTension,
    ActionUnitScore,
    HealthStatus,
    evaluate_cat_health
)

__all__ = [
    "onto",
    "Cat",
    "FacialFeature",
    "EarPosition",
    "OrbitalTightening",
    "MuzzleTension",
    "ActionUnitScore",
    "HealthStatus",
    "evaluate_cat_health"
]
