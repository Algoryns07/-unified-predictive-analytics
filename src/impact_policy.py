"""Classify measured impact and make a transparent retraining recommendation."""
def decide(drift_detected, reference_f1, current_f1, harmful_drop=.05,
           severe_drop=.12, retraining_cost=.08, minimum_benefit=.03):
    drop = float(reference_f1 - current_f1)
    if not drift_detected:
        label = "NO_DRIFT"
        action = "MONITOR"
    elif drop >= harmful_drop:
        label = "DRIFT_HARMFUL"
        action = "RETRAIN" if drop >= max(severe_drop, retraining_cost + minimum_benefit) else "INVESTIGATE"
    else:
        label = "DRIFT_NO_HARM"
        action = "MONITOR"
    return {"impact_class": label, "recommended_action": action, "f1_drop": drop}
