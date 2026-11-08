from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _validate_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(value)


def _validate_threshold(value, name, lower, upper, strict_lower=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    valid_lower = value > lower if strict_lower else value >= lower

    if not valid_lower or value > upper:
        raise ValueError(f"Invalid value for {name}.")

    return value


def _validate_frames(reference, current):
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("reference must be a pandas DataFrame.")

    if not isinstance(current, pd.DataFrame):
        raise TypeError("current must be a pandas DataFrame.")

    if reference.empty or current.empty:
        raise ValueError("Both datasets must contain observations.")

    if not reference.columns.is_unique or not current.columns.is_unique:
        raise ValueError("Feature names must be unique.")

    if set(reference.columns) != set(current.columns):
        raise ValueError("Both datasets must contain the same feature columns.")

    current = current.loc[:, reference.columns]

    for name, frame in [
        ("reference", reference),
        ("current", current)
    ]:
        for column in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise TypeError(f"{name}[{column!r}] must be numeric.")

            values = frame[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{name}[{column!r}] contains non-finite values."
                )

    return reference.astype(float), current.astype(float)


def population_stability_index(
    reference,
    current,
    bins=10,
    epsilon=1e-6
):
    """Calculate PSI with fallback bins for degenerate distributions."""

    bins = _validate_integer(bins, "bins", minimum=2)

    epsilon = _validate_threshold(
        epsilon,
        "epsilon",
        lower=0.0,
        upper=0.1,
        strict_lower=True
    )

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.ndim != 1 or current.ndim != 1:
        raise ValueError("PSI inputs must be one-dimensional.")

    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI inputs must not be empty.")

    if not np.isfinite(reference).all():
        raise ValueError("Reference values must be finite.")

    if not np.isfinite(current).all():
        raise ValueError("Current values must be finite.")

    # Normally, bin boundaries are learned from the reference.
    # If the reference is constant, use pooled observations so that
    # a shifted current distribution can still be distinguished.
    if np.unique(reference).size < 2:
        pooled = np.concatenate((reference, current))

        if np.unique(pooled).size < 2:
            return 0.0

        edges = np.unique(
            np.quantile(
                pooled,
                np.linspace(0, 1, bins + 1)
            )
        )
    else:
        edges = np.unique(
            np.quantile(
                reference,
                np.linspace(0, 1, bins + 1)
            )
        )

    # Two edges alone would create only one histogram bin after
    # replacing the endpoints with infinities. Add a midpoint.
    if edges.size == 2:
        midpoint = edges[0] + (edges[1] - edges[0]) / 2
        edges = np.array([edges[0], midpoint, edges[1]])

    if edges.size < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts.astype(float) / reference.size
    cur_pct = cur_counts.astype(float) / current.size

    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi = np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    )

    return float(psi)




def benjamini_hochberg(p_values):
    """Calculate Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence.")

    if not np.isfinite(values).all():
        raise ValueError("p_values must be finite.")

    if np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must lie between 0 and 1.")

    count = values.size
    order = np.argsort(values)

    ranked = values[order] * count / np.arange(1, count + 1)

    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def detect_feature_drift(
    reference,
    current,
    psi_threshold=0.20,
    q_value_threshold=0.05,
    bins=10,
    min_drifted_features=2
):
    """
    Compare numeric feature distributions.

    A feature is flagged when:
        PSI >= psi_threshold OR BH-adjusted KS q-value <= q_value_threshold.

    An overall alert requires at least min_drifted_features flagged features.

    This is a checkpoint-level drift signal. Sustained-trigger logic and
    retraining decisions belong to the calling experiment.
    """

    psi_threshold = _validate_threshold(
        psi_threshold, "psi_threshold", 0.0, math.inf
    )

    q_value_threshold = _validate_threshold(
        q_value_threshold, "q_value_threshold", 0.0, 1.0
    )

    bins = _validate_integer(bins, "bins", minimum=2)

    min_drifted_features = _validate_integer(
        min_drifted_features,
        "min_drifted_features",
        minimum=1
    )

    reference, current = _validate_frames(reference, current)

    rows = []
    p_values = []

    for feature in reference.columns:

        ref_values = reference[feature].to_numpy()
        cur_values = current[feature].to_numpy()

        psi = population_stability_index(
            ref_values,
            cur_values,
            bins=bins
        )

        ks_result = ks_2samp(
            ref_values,
            cur_values,
            alternative="two-sided",
            method="auto"
        )

        p_values.append(float(ks_result.pvalue))

        rows.append({
            "feature": str(feature),
            "reference_observations": len(ref_values),
            "current_observations": len(cur_values),
            "psi": psi,
            "ks_statistic": float(ks_result.statistic),
            "ks_p_value": float(ks_result.pvalue)
        })

    q_values = benjamini_hochberg(p_values)

    results = pd.DataFrame(rows)

    results["ks_q_value"] = q_values

    results["psi_drift"] = (
        results["psi"] >= psi_threshold
    )

    results["ks_drift"] = (
        results["ks_q_value"] <= q_value_threshold
    )

    results["feature_drift"] = (
        results["psi_drift"] | results["ks_drift"]
    )

    drifted_features = results.loc[
        results["feature_drift"],
        "feature"
    ].tolist()

    drift_count = len(drifted_features)

    return {
        "feature_results": results,
        "drift_detected": drift_count >= min_drifted_features,
        "drifted_feature_count": drift_count,
        "drifted_features": drifted_features,
        "features_tested": len(results),
        "psi_threshold": psi_threshold,
        "q_value_threshold": q_value_threshold,
        "min_drifted_features": min_drifted_features,
        "correction": "Benjamini-Hochberg"
    }
