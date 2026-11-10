from src.impact_policy import decide

def test_non_harmful_drift_monitor():
    result = decide(True, .90, .89)
    assert result["impact_class"] == "DRIFT_NO_HARM"
    assert result["recommended_action"] == "MONITOR"

def test_severe_harm_retrain():
    result = decide(True, .95, .70)
    assert result["recommended_action"] == "RETRAIN"
