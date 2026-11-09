"""Reusable upload-time dataset profiling, regression, and feature-drift utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def profile_dataset(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        raise ValueError("The uploaded dataset is empty.")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    return {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "numeric_columns": numeric, "categorical_columns": categorical,
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_by_column": frame.isna().sum().sort_values(ascending=False),
    }


def _prepare_features(reference: pd.DataFrame, incoming: pd.DataFrame, target: str):
    if target not in reference.columns:
        raise ValueError(f"Target '{target}' is not present in the reference dataset.")
    features = [c for c in reference.columns if c != target and c in incoming.columns]
    if not features:
        raise ValueError("No shared input features remain after excluding the target.")
    # Retain columns that have at least one usable value in reference data.
    features = [c for c in features if not reference[c].isna().all()]
    if not features:
        raise ValueError("All candidate input features are empty in the reference dataset.")
    return features


def train_and_predict(reference: pd.DataFrame, incoming: pd.DataFrame, target: str,
                      model_name: str = "HistGradientBoosting", test_fraction: float = 0.2):
    """Chronologically split reference rows; fit and predict incoming rows."""
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("Test fraction must be between 0.1 and 0.5.")
    features = _prepare_features(reference, incoming, target)
    train_frame = reference[features + [target]].copy()
    train_frame[target] = pd.to_numeric(train_frame[target], errors="coerce")
    train_frame = train_frame.dropna(subset=[target])
    if len(train_frame) < 20:
        raise ValueError("At least 20 reference rows with a valid target are required.")

    # If a parseable timestamp exists, sort chronologically before splitting.
    timestamp_candidates = [c for c in features if c.lower() in {"timestamp", "datetime", "date", "time"}]
    if timestamp_candidates:
        tcol = timestamp_candidates[0]
        parsed = pd.to_datetime(train_frame[tcol], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            train_frame = train_frame.assign(_sort_time=parsed).sort_values("_sort_time").drop(columns="_sort_time")

    split = max(1, min(len(train_frame)-1, int(len(train_frame) * (1-test_fraction))))
    fit = train_frame.iloc[:split]
    holdout = train_frame.iloc[split:]
    numeric = fit[features].select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=3, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=0.08, min_samples_leaf=20, l2_regularization=1.0, random_state=42),
    }
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
    pipeline = Pipeline([("preprocess", preprocess), ("model", models[model_name])])
    pipeline.fit(fit[features], fit[target])
    holdout_pred = pipeline.predict(holdout[features])
    y_holdout = holdout[target].to_numpy(dtype=float)
    metrics = {
        "MAE": float(mean_absolute_error(y_holdout, holdout_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_holdout, holdout_pred))),
        "R2": float(r2_score(y_holdout, holdout_pred)) if len(y_holdout) > 1 else float("nan"),
        "training_rows": int(len(fit)), "validation_rows": int(len(holdout)),
    }
    incoming_features = incoming.reindex(columns=features)
    predictions = pipeline.predict(incoming_features)
    result = incoming.copy()
    result["prediction"] = predictions
    if target in incoming.columns:
        actual = pd.to_numeric(incoming[target], errors="coerce")
        valid = actual.notna()
        result["prediction_error"] = np.where(valid, actual - predictions, np.nan)
        if valid.sum() > 0:
            metrics["incoming_MAE"] = float(mean_absolute_error(actual[valid], predictions[valid]))
            metrics["incoming_RMSE"] = float(np.sqrt(mean_squared_error(actual[valid], predictions[valid])))
            metrics["incoming_labeled_rows"] = int(valid.sum())
    return result, metrics, features


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        pooled = np.concatenate([reference, current])
        edges = np.unique(np.quantile(pooled, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins=edges)[0].astype(float) / len(reference)
    b = np.histogram(current, bins=edges)[0].astype(float) / len(current)
    eps = 1e-6
    a = np.clip(a, eps, None); b = np.clip(b, eps, None)
    a /= a.sum(); b /= b.sum()
    return float(np.sum((b-a) * np.log(b/a)))


def _bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    ranked = p[order] * n / np.arange(1, n+1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    return adjusted


def compare_feature_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in reference or feature not in current:
            continue
        a = pd.to_numeric(reference[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        b = pd.to_numeric(current[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if len(a) < 2 or len(b) < 2:
            continue
        ks = ks_2samp(a, b, alternative="two-sided", method="auto")
        rows.append({"feature": feature, "reference_n": len(a), "incoming_n": len(b),
                     "PSI": _psi(a, b), "KS_statistic": float(ks.statistic),
                     "KS_p_value": float(ks.pvalue),
                     "Wasserstein": float(wasserstein_distance(a, b)),
                     "reference_mean": float(np.mean(a)), "incoming_mean": float(np.mean(b))})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["adjusted_q_value"] = _bh_adjust(result["KS_p_value"].to_numpy())
        result["drift_flag"] = (result["PSI"] >= 0.2) | (result["adjusted_q_value"] < 0.05)
        result = result.sort_values(["drift_flag", "PSI"], ascending=[False, False]).reset_index(drop=True)
    return result

from math import ceil, isfinite


def recommend_retraining(
    drift: pd.DataFrame,
    metrics: dict,
    performance_threshold: float = 0.25,
    minimum_labeled_rows: int = 30,
) -> dict:
    """
    Produce an evidence-based retraining screening recommendation.

    Feature drift alone never triggers a retraining recommendation.
    Sufficient labelled incoming observations are required to assess
    measured prediction-performance degradation.
    """
    if not 0 <= performance_threshold <= 5:
        raise ValueError("Performance threshold must be between 0 and 5.")

    if minimum_labeled_rows < 1:
        raise ValueError("Minimum labelled rows must be positive.")

    drifted_features = 0

    if drift is not None and not drift.empty:
        if "drift_flag" in drift.columns:
            drifted_features = int(
                drift["drift_flag"].fillna(False).astype(bool).sum()
            )

    has_drift = drifted_features > 0

    labeled_rows = int(metrics.get("incoming_labeled_rows", 0) or 0)

    baseline_mae = metrics.get("MAE")
    incoming_mae = metrics.get("incoming_MAE")

    valid_metrics = (
        baseline_mae is not None
        and incoming_mae is not None
        and isfinite(float(baseline_mae))
        and isfinite(float(incoming_mae))
        and float(baseline_mae) > 0
    )

    performance_verified = (
        valid_metrics and labeled_rows >= minimum_labeled_rows
    )

    degradation_ratio = None
    performance_degraded = False

    if performance_verified:
        degradation_ratio = (
            float(incoming_mae) / float(baseline_mae)
        ) - 1.0

        performance_degraded = (
            degradation_ratio > performance_threshold
        )

    if performance_degraded:
        status = "RETRAIN_CANDIDATE"
        recommendation = (
            "Evaluate a challenger model before approving retraining."
        )
        reason = (
            "Measured incoming prediction error exceeds the reference "
            "validation error by more than the configured threshold."
        )

    elif has_drift:
        status = "MONITOR"
        recommendation = (
            "Continue monitoring and collect labelled observations."
        )

        if not performance_verified:
            reason = (
                "Feature drift is present, but there are insufficient "
                "labelled observations to verify performance degradation."
            )
        else:
            reason = (
                "Feature drift is present, but measured performance "
                "degradation has not crossed the configured threshold."
            )

    else:
        status = "CONTINUE"
        recommendation = (
            "Continue using the current model and monitor future data."
        )

        if not performance_verified:
            reason = (
                "No monitored feature crossed the drift threshold. "
                "Incoming performance is not fully verified."
            )
        else:
            reason = (
                "Neither monitored feature drift nor significant "
                "performance degradation was detected."
            )

    return {
        "status": status,
        "recommendation": recommendation,
        "reason": reason,
        "drifted_features": drifted_features,
        "performance_verified": performance_verified,
        "performance_degraded": performance_degraded,
        "labeled_rows": labeled_rows,
        "minimum_labeled_rows": minimum_labeled_rows,
        "baseline_MAE": float(baseline_mae) if valid_metrics else None,
        "incoming_MAE": float(incoming_mae) if valid_metrics else None,
        "MAE_degradation_ratio": degradation_ratio,
        "performance_threshold": performance_threshold,
    }


def estimate_retraining_costs(
    decision: dict,
    horizon_windows: int = 30,
    periodic_interval: int = 7,
    retraining_cost: float = 2500.0,
    monitoring_cost_per_window: float = 50.0,
) -> dict:
    """
    Compare estimated periodic and impact-aware operational costs.

    Costs are user-supplied assumptions, not measured financial savings.
    The comparison excludes prediction-error impact, labour not included
    in the assumptions, downtime and infrastructure overhead.
    """
    if horizon_windows < 1:
        raise ValueError("Horizon must be at least one window.")

    if periodic_interval < 1:
        raise ValueError("Periodic interval must be positive.")

    if retraining_cost < 0 or monitoring_cost_per_window < 0:
        raise ValueError("Costs cannot be negative.")

    periodic_events = ceil(horizon_windows / periodic_interval)

    adaptive_events = (
        1 if decision.get("status") == "RETRAIN_CANDIDATE" else 0
    )

    monitoring_total = (
        horizon_windows * monitoring_cost_per_window
    )

    periodic_retraining_total = periodic_events * retraining_cost
    adaptive_retraining_total = adaptive_events * retraining_cost

    periodic_total = periodic_retraining_total + monitoring_total
    adaptive_total = adaptive_retraining_total + monitoring_total

    savings = periodic_total - adaptive_total

    savings_percentage = (
        100.0 * savings / periodic_total
        if periodic_total > 0 else 0.0
    )

    return {
        "horizon_windows": horizon_windows,
        "periodic_retraining_events": periodic_events,
        "adaptive_retraining_events": adaptive_events,
        "monitoring_cost": monitoring_total,
        "periodic_retraining_cost": periodic_retraining_total,
        "adaptive_retraining_cost": adaptive_retraining_total,
        "periodic_total_cost": periodic_total,
        "adaptive_total_cost": adaptive_total,
        "estimated_savings": savings,
        "estimated_savings_percentage": savings_percentage,
        "currency": "INR",
    }
