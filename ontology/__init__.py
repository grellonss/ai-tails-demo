"""
Ontology module for AI-Tails: Feline Health ontology (FGS + temperature) and SWRL rules.
"""

from .feline_health_ontology import (
    ActionUnitScore, Cat, EarPosition, FacialFeature, HealthStatus, MuzzleTension, OrbitalTightening,
    ThermalStatus, evaluate_cat_health, onto,
)
from .rules import RULES, SEVERITY

__all__ = [
    "onto", "Cat", "FacialFeature", "EarPosition", "OrbitalTightening", "MuzzleTension",
    "ActionUnitScore", "HealthStatus", "ThermalStatus", "evaluate_cat_health", "RULES", "SEVERITY",
]
