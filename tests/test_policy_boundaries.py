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
