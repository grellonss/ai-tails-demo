"""
SWRL rules of the Feline Health ontology, in a single table.

Each rule has: the SWRL text run by Pellet, the health status it assigns and an equivalent
Python predicate, used only as a fallback if Pellet/Java is not available. Keeping text and
predicate on the same row prevents the two versions from diverging.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Rule:
    name: str
    status: str                                            # HealthStatus individual assigned
    swrl: str
    holds: Callable[[int, Optional[float]], bool]          # (total FGS score, temperature) -> bool


RULES = [
    # --- Feline Grimace Scale (face only) ------------------------------------------
    Rule("fgs_good", "Good",
         "Cat(?c) ^ hasTotalScore(?c, ?s) ^ lessThanOrEqual(?s, 1) -> hasHealthStatus(?c, Good)",
         lambda s, t: s <= 1),
    Rule("fgs_medium", "Medium",
         "Cat(?c) ^ hasTotalScore(?c, ?s) ^ greaterThanOrEqual(?s, 2) ^ lessThanOrEqual(?s, 3) "
         "-> hasHealthStatus(?c, Medium)",
         lambda s, t: 2 <= s <= 3),
    Rule("fgs_poor", "Poor",
         "Cat(?c) ^ hasTotalScore(?c, ?s) ^ greaterThanOrEqual(?s, 4) -> hasHealthStatus(?c, Poor)",
         lambda s, t: s >= 4),
    # --- Multimodal: face + temperature --------------------------------------------
    # Marked FGS + fever: squinted eyes are not sleep, but an active inflammatory process.
    Rule("fgs_fever_acute", "AcuteInflammatoryPain",
         "Cat(?c) ^ hasTotalScore(?c, ?s) ^ greaterThanOrEqual(?s, 4) ^ hasTemperature(?c, ?t) "
         "^ greaterThanOrEqual(?t, 39.5) -> hasHealthStatus(?c, AcuteInflammatoryPain)",
         lambda s, t: s >= 4 and t is not None and t >= 39.5),
]

# From least to most severe: the primary status shown to the user is the most severe one inferred.
SEVERITY = ["Good", "Medium", "Poor", "AcuteInflammatoryPain"]
