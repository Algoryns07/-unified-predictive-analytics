import numpy as np


def calculate_error_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Inputs must be non-empty and equally sized.")

    if not np.isfinite(actual).all():
        raise ValueError("Actual values must be finite.")

    if not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite.")

    residuals = actual - predicted

    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "bias": float(np.mean(residuals)),
    }


def evaluate_performance_deterioration(
    window_mae,
    baseline_mae,
    threshold_multiplier=1.5,
):

    if not np.isfinite(window_mae) or window_mae < 0:
        raise ValueError("Window MAE must be finite and non-negative.")

    if not np.isfinite(baseline_mae) or baseline_mae <= 0:
        raise ValueError("Baseline MAE must be finite and positive.")

    if (
        isinstance(threshold_multiplier, bool)
        or not isinstance(threshold_multiplier, (int, float))
        or not np.isfinite(threshold_multiplier)
        or threshold_multiplier <= 1
    ):
        raise ValueError("Threshold multiplier must be greater than 1.")

    threshold = baseline_mae * threshold_multiplier

    return {
        "threshold": float(threshold),
        "deteriorated": bool(window_mae > threshold),
        "error_ratio": float(window_mae / baseline_mae),
    }


def identify_sustained_deterioration(
    flags,
    block_ids,
    required_windows=2,
):

    flags = list(flags)
    block_ids = list(block_ids)

    if len(flags) != len(block_ids):
        raise ValueError("Flags and block IDs must have equal lengths.")

    if (
        isinstance(required_windows, bool)
        or not isinstance(required_windows, int)
        or required_windows < 1
    ):
        raise ValueError("required_windows must be a positive integer.")

    sustained = [False] * len(flags)

    for index in range(len(flags)):

        if not flags[index]:
            continue

        block = block_ids[index]
        count = 0

        for previous in range(index, -1, -1):

            if block_ids[previous] != block:
                break

            if not flags[previous]:
                break

            count += 1

        sustained[index] = count >= required_windows

    return sustained


def calculate_signal_association(
    drift_flags,
    performance_flags,
):

    drift = np.asarray(drift_flags, dtype=bool)
    performance = np.asarray(performance_flags, dtype=bool)

    if len(drift) == 0 or len(drift) != len(performance):
        raise ValueError("Signals must be non-empty and equally sized.")

    both = int(np.sum(drift & performance))
    drift_only = int(np.sum(drift & ~performance))
    performance_only = int(np.sum(~drift & performance))
    neither = int(np.sum(~drift & ~performance))

    total = len(drift)

    return {
        "windows": int(total),
        "both": both,
        "drift_only": drift_only,
        "performance_only": performance_only,
        "neither": neither,
        "cooccurrence_rate_pct": float(100 * both / total),
        "performance_rate_pct": float(100 * np.mean(performance)),
        "drift_rate_pct": float(100 * np.mean(drift)),
    }
