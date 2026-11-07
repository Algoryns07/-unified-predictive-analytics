"""Classify measured impact and make a transparent retraining recommendation."""

import math


def decide(
    drift_detected,
    reference_f1,
    current_f1,
    harmful_drop=0.05,
    severe_drop=0.12,
    retraining_cost=0.08,
    minimum_benefit=0.03
):
    """Recommend an action using validated F1 and cost thresholds."""

    scores = {
        "reference_f1": reference_f1,
        "current_f1": current_f1,
    }

    for name, value in scores.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    thresholds = {
        "harmful_drop": harmful_drop,
        "severe_drop": severe_drop,
        "retraining_cost": retraining_cost,
        "minimum_benefit": minimum_benefit,
    }

    for name, value in thresholds.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")

    if severe_drop < harmful_drop:
        raise ValueError("severe_drop must be >= harmful_drop.")

    if not isinstance(drift_detected, bool):
        raise ValueError("drift_detected must be boolean.")

    drop = float(reference_f1 - current_f1)

    if not drift_detected:
        label = "NO_DRIFT"
        action = "MONITOR"
    elif drop >= harmful_drop:
        label = "DRIFT_HARMFUL"
        action = (
            "RETRAIN"
            if drop >= max(severe_drop, retraining_cost + minimum_benefit)
            else "INVESTIGATE"
        )
    else:
        label = "DRIFT_NO_HARM"
        action = "MONITOR"

    return {
        "impact_class": label,
        "recommended_action": action,
        "f1_drop": drop
    }
