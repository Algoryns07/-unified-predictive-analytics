from src.impact_policy import decide


def test_no_drift_never_triggers_retraining():
    result = decide(False, 0.95, 0.50)
    assert result["recommended_action"] == "MONITOR"


def test_small_performance_drop_triggers_monitoring():
    result = decide(True, 0.90, 0.88)
    assert result["recommended_action"] == "MONITOR"


def test_moderate_harm_triggers_investigation():
    result = decide(True, 0.95, 0.88)
    assert result["recommended_action"] == "INVESTIGATE"


def test_severe_harm_triggers_retraining():
    result = decide(True, 0.95, 0.80)
    assert result["recommended_action"] == "RETRAIN"


def test_performance_improvement_does_not_trigger_retraining():
    result = decide(True, 0.85, 0.92)
    assert result["recommended_action"] == "MONITOR"

import pytest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_f1": 1.2},
        {"current_f1": -0.1},
        {"harmful_drop": -0.1},
        {"severe_drop": 0.01, "harmful_drop": 0.05},
        {"retraining_cost": float("nan")},
    ],
)
def test_invalid_policy_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        decide(**{"drift_detected": True, "reference_f1": 0.95, "current_f1": 0.80, **kwargs})


def test_policy_rejects_non_boolean_drift_flag():
    with pytest.raises(ValueError):
        decide(1, 0.95, 0.80)
