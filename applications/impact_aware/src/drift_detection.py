import numbers

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from statsmodels.stats.multitest import multipletests


def _validate_threshold(name, value, minimum=None, maximum=None):
    if not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")

    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")


def psi(reference, current, bins=10, epsilon=1e-6):
    if isinstance(bins, bool) or not isinstance(bins, numbers.Integral) or bins < 2:
        raise ValueError("bins must be an integer >= 2.")

    _validate_threshold("epsilon", epsilon, minimum=0)
    if epsilon <= 0 or epsilon >= 0.5:
        raise ValueError("epsilon must be strictly between 0 and 0.5.")

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    a = np.histogram(ref, bins=edges)[0].astype(float)
    b = np.histogram(cur, bins=edges)[0].astype(float)

    a = np.clip(a / a.sum(), epsilon, None)
    b = np.clip(b / b.sum(), epsilon, None)

    return float(np.sum((b - a) * np.log(b / a)))


def compare_feature(reference, current):
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return {
            "psi": 0.0,
            "ks_statistic": 0.0,
            "ks_p_value": 1.0,
            "wasserstein": 0.0,
            "normalized_wasserstein": 0.0
        }

    ks = ks_2samp(ref, cur)
    raw_distance = wasserstein_distance(ref, cur)

    scale = np.std(ref)
    normalized_distance = (
        float(raw_distance / scale) if scale > 0 else float(raw_distance)
    )

    return {
        "psi": psi(ref, cur),
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "wasserstein": float(raw_distance),
        "normalized_wasserstein": normalized_distance
    }


def compare_frames(
    reference,
    current,
    psi_threshold=0.10,
    ks_alpha=0.05,
    wasserstein_threshold=0.20
):
    _validate_threshold("psi_threshold", psi_threshold, minimum=0)
    _validate_threshold("ks_alpha", ks_alpha, minimum=0, maximum=1)
    if ks_alpha == 0 or ks_alpha == 1:
        raise ValueError("ks_alpha must be strictly between 0 and 1.")

    _validate_threshold(
        "wasserstein_threshold", wasserstein_threshold, minimum=0
    )

    if not isinstance(reference, pd.DataFrame):
        raise TypeError("reference must be a pandas DataFrame.")

    if not isinstance(current, pd.DataFrame):
        raise TypeError("current must be a pandas DataFrame.")

    common_columns = reference.columns.intersection(current.columns)
    result = {}

    for col in common_columns:
        result[col] = compare_feature(reference[col], current[col])

    if not result:
        return {
            "features": {},
            "drifted_features": [],
            "any_drift": False
        }

    columns = list(result.keys())
    p_values = [result[col]["ks_p_value"] for col in columns]

    rejected, adjusted_p_values, _, _ = multipletests(
        p_values,
        alpha=ks_alpha,
        method="fdr_bh"
    )

    for i, col in enumerate(columns):
        stats = result[col]
        stats["ks_adjusted_p_value"] = float(adjusted_p_values[i])

        stats["drift"] = bool(
            stats["psi"] >= psi_threshold
            or rejected[i]
            or stats["normalized_wasserstein"] >= wasserstein_threshold
        )

    drifted_features = [
        col for col, stats in result.items() if stats["drift"]
    ]

    return {
        "features": result,
        "drifted_features": drifted_features,
        "any_drift": bool(drifted_features)
    }
