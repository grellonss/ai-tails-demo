import pytest

from ontology import evaluate_cat_health, onto

CASES = [
    # (ear, eye, muzzle, temperature, thermal status) -> primary status, fired rules
    ((0, 0, 1, 38.6, "Normothermia"), "Good", ["fgs_good"]),
    ((1, 1, 1, 38.6, "Normothermia"), "Medium", ["fgs_medium"]),
    ((2, 1, 1, 38.9, "Normothermia"), "Poor", ["fgs_poor"]),
    ((2, 2, 1, 39.8, "MildFever"), "AcuteInflammatoryPain", ["fgs_poor", "fgs_fever_acute"]),
    ((2, 2, 2, 39.4, "MildFever"), "Poor", ["fgs_poor"]),          # below the 39.5 °C threshold
    ((0, 0, 0, None, None), "Good", ["fgs_good"]),                   # without temperature
]


@pytest.mark.parametrize("args, primary, rules", CASES)
@pytest.mark.parametrize("run_reasoner", [True, False], ids=["pellet", "fallback"])
def test_rules(args, primary, rules, run_reasoner):
    r = evaluate_cat_health("Oliver", *args, run_reasoner=run_reasoner)
    assert r["primary_status"] == primary
    assert [x["name"] for x in r["fired_rules"]] == rules
    assert r["reasoner"] == ("pellet" if run_reasoner else "python-fallback")


def test_pellet_is_really_used():
    """If this fails, Java/Owlready2 are not compatible (see README: owlready2==0.47 with Java 21)."""
    assert evaluate_cat_health("Oliver", 0, 0, 0)["reasoner"] == "pellet"


def test_individuals_do_not_accumulate():
    evaluate_cat_health("Oliver", 2, 2, 2, 40.1, "HighFever")
    r = evaluate_cat_health("Oliver", 0, 0, 0, 38.5, "Normothermia")
    assert r["inferred_status"] == ["Good"]
    assert list(onto.Cat.instances()) == []


def test_invalid_input():
    with pytest.raises(ValueError):
        evaluate_cat_health("Oliver", 3, 0, 0)
    with pytest.raises(ValueError):
        evaluate_cat_health("Oliver", 0, 0, 0, 38.5, "Boiling")
