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
from sklearn.inspection import permutation_importance


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
    # Predictive association on the held-out reference window; not causal evidence.
    try:
        importance = permutation_importance(
            pipeline, holdout[features], holdout[target],
            scoring="neg_mean_absolute_error", n_repeats=3,
            random_state=42, n_jobs=1,
        )
        metrics["feature_importance"] = {
            feature: float(max(0.0, value))
            for feature, value in zip(features, importance.importances_mean)
        }
    except Exception:
        metrics["feature_importance"] = {}
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


def analyze_dataset_shift(reference: pd.DataFrame, incoming: pd.DataFrame) -> dict:
    # Independent all-column comparison; does not depend on the selected target.
    if reference is None or incoming is None or reference.empty or incoming.empty:
        raise ValueError("Both datasets must contain at least one row.")
    if not reference.columns.is_unique or not incoming.columns.is_unique:
        raise ValueError("Column names must be unique.")

    ref_cols, inc_cols = list(reference.columns), list(incoming.columns)
    ref_set, inc_set = set(ref_cols), set(inc_cols)
    rows = []

    for col in ref_cols + [c for c in inc_cols if c not in ref_set]:
        in_ref, in_inc = col in ref_set, col in inc_set
        if not in_ref or not in_inc:
            x = reference[col] if in_ref else None
            y = incoming[col] if in_inc else None
            rows.append({
                "feature": col, "analysis_type": "Schema",
                "schema_status": "Missing from incoming" if in_ref else "New in incoming",
                "reference_dtype": str(x.dtype) if in_ref else None,
                "incoming_dtype": str(y.dtype) if in_inc else None,
                "reference_missing_pct": float(x.isna().mean()*100) if in_ref else np.nan,
                "incoming_missing_pct": float(y.isna().mean()*100) if in_inc else np.nan,
                "reference_unique": int(x.nunique()) if in_ref else 0,
                "incoming_unique": int(y.nunique()) if in_inc else 0,
                "PSI": np.nan, "KS_statistic": np.nan, "KS_p_value": np.nan,
                "Wasserstein": np.nan, "TV_distance": np.nan, "test_p_value": np.nan,
                "reference_mean": np.nan, "incoming_mean": np.nan,
                "reference_median": np.nan, "incoming_median": np.nan,
                "reference_std": np.nan, "incoming_std": np.nan,
                "reference_min": np.nan, "incoming_min": np.nan,
                "reference_max": np.nan, "incoming_max": np.nan,
                "reference_nonfinite_count": np.nan, "incoming_nonfinite_count": np.nan,
                "drift_flag": True, "severity": "High",
                "recommended_action": "Review schema compatibility",
            })
            continue

        x, y = reference[col], incoming[col]
        changed_type = str(x.dtype) != str(y.dtype)
        base = {
            "feature": col, "reference_dtype": str(x.dtype), "incoming_dtype": str(y.dtype),
            "schema_status": "Type changed" if changed_type else "Unchanged",
            "reference_missing_pct": float(x.isna().mean()*100),
            "incoming_missing_pct": float(y.isna().mean()*100),
            "reference_unique": int(x.nunique()), "incoming_unique": int(y.nunique()),
        }

        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x_raw = pd.to_numeric(x, errors="coerce")
            y_raw = pd.to_numeric(y, errors="coerce")
            x_values = x_raw.to_numpy(dtype=float, na_value=np.nan)
            y_values = y_raw.to_numpy(dtype=float, na_value=np.nan)
            reference_nonfinite = int((~np.isfinite(x_values) & ~np.isnan(x_values)).sum())
            incoming_nonfinite = int((~np.isfinite(y_values) & ~np.isnan(y_values)).sum())
            xn = x_raw.replace([np.inf,-np.inf], np.nan).dropna().to_numpy(float)
            yn = y_raw.replace([np.inf,-np.inf], np.nan).dropna().to_numpy(float)
            if len(xn) >= 2 and len(yn) >= 2:
                ks = ks_2samp(xn, yn, alternative="two-sided", method="auto")
                psi, wd = _psi(xn, yn), float(wasserstein_distance(xn, yn))
                drift = bool(psi >= .2)
                severity = "High" if psi >= .5 else ("Moderate" if drift else "Low")
                rows.append({
                    **base, "analysis_type": "Numeric", "PSI": psi,
                    "KS_statistic": float(ks.statistic), "KS_p_value": float(ks.pvalue),
                    "Wasserstein": wd, "TV_distance": np.nan, "test_p_value": float(ks.pvalue),
                    "reference_mean": float(np.mean(xn)), "incoming_mean": float(np.mean(yn)),
                    "reference_median": float(np.median(xn)), "incoming_median": float(np.median(yn)),
                    "reference_std": float(np.std(xn)), "incoming_std": float(np.std(yn)),
                    "reference_min": float(np.min(xn)), "incoming_min": float(np.min(yn)),
                    "reference_max": float(np.max(xn)), "incoming_max": float(np.max(yn)),
                    "reference_nonfinite_count": reference_nonfinite,
                    "incoming_nonfinite_count": incoming_nonfinite,
                    "drift_flag": drift, "severity": severity,
                    "recommended_action": "Investigate numeric distribution shift" if drift else "Routine monitoring",
                })
            else:
                rows.append({
                    **base, "analysis_type": "Numeric", "PSI": np.nan,
                    "KS_statistic": np.nan, "KS_p_value": np.nan, "Wasserstein": np.nan,
                    "TV_distance": np.nan, "test_p_value": np.nan,
                    "reference_mean": np.nan, "incoming_mean": np.nan,
                    "reference_median": np.nan, "incoming_median": np.nan,
                    "reference_std": np.nan, "incoming_std": np.nan,
                    "reference_min": np.nan, "incoming_min": np.nan,
                    "reference_max": np.nan, "incoming_max": np.nan,
                    "reference_nonfinite_count": reference_nonfinite,
                    "incoming_nonfinite_count": incoming_nonfinite,
                    "drift_flag": False, "severity": "Insufficient data",
                    "recommended_action": "Collect at least two valid values per dataset",
                })
        else:
            xs = x.astype("string").fillna("<MISSING>")
            ys = y.astype("string").fillna("<MISSING>")
            top = set(xs.value_counts().head(100).index) | set(ys.value_counts().head(100).index)
            xs, ys = xs.where(xs.isin(top), "<OTHER>"), ys.where(ys.isin(top), "<OTHER>")
            cats = sorted(set(xs.unique()) | set(ys.unique()))
            xp = xs.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
            yp = ys.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
            tv = float(.5 * np.abs(xp-yp).sum())
            table = np.vstack([
                xs.value_counts().reindex(cats, fill_value=0).to_numpy(),
                ys.value_counts().reindex(cats, fill_value=0).to_numpy(),
            ])
            pv = np.nan
            if len(cats) > 1:
                try:
                    from scipy.stats import chi2_contingency
                    _, pv, _, _ = chi2_contingency(table, correction=False)
                    pv = float(pv)
                except Exception:
                    pv = np.nan
            drift = bool(tv >= .2)
            severity = "High" if tv >= .5 else ("Moderate" if drift else "Low")
            rows.append({
                **base, "analysis_type": "Categorical", "PSI": np.nan,
                "KS_statistic": np.nan, "KS_p_value": np.nan, "Wasserstein": np.nan,
                "TV_distance": tv, "test_p_value": pv,
                "reference_mean": np.nan, "incoming_mean": np.nan,
                "reference_median": np.nan, "incoming_median": np.nan,
                "reference_std": np.nan, "incoming_std": np.nan,
                "reference_min": np.nan, "incoming_min": np.nan,
                "reference_max": np.nan, "incoming_max": np.nan,
                "reference_nonfinite_count": np.nan, "incoming_nonfinite_count": np.nan,
                "drift_flag": drift, "severity": severity,
                "recommended_action": "Review category mix and unseen values" if drift else "Routine monitoring",
            })

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["adjusted_q_value"] = np.nan
        pvals = pd.to_numeric(frame["test_p_value"], errors="coerce")
        valid = pvals.notna()
        if valid.any():
            frame.loc[valid, "adjusted_q_value"] = _bh_adjust(pvals[valid].to_numpy())
        numeric_effect = pd.to_numeric(frame["PSI"], errors="coerce").ge(0.2)
        categorical_effect = pd.to_numeric(frame["TV_distance"], errors="coerce").ge(0.2)
        statistical_significance = pd.to_numeric(frame["adjusted_q_value"], errors="coerce").lt(0.05)
        statistical_rows = frame["analysis_type"].isin(["Numeric", "Categorical"])
        statistical_drift = statistical_rows & (numeric_effect.fillna(False) | categorical_effect.fillna(False) | statistical_significance.fillna(False))
        frame.loc[statistical_rows, "drift_flag"] = statistical_drift.loc[statistical_rows]
        frame.loc[statistical_rows & statistical_drift, "severity"] = "Moderate"
        frame.loc[statistical_rows & ~statistical_drift, "severity"] = "Low"
        frame.loc[statistical_rows & statistical_drift, "recommended_action"] = "Investigate statistically adjusted distribution shift"
        frame.loc[statistical_rows & ~statistical_drift, "recommended_action"] = "Routine monitoring"
        frame["missingness_change_pp"] = frame["incoming_missing_pct"] - frame["reference_missing_pct"]
        missing_change = frame["missingness_change_pp"].abs() >= 10
        type_change = frame["schema_status"].ne("Unchanged")
        frame.loc[missing_change | type_change, "drift_flag"] = True
        frame.loc[missing_change, "severity"] = "Moderate"
        frame.loc[missing_change, "recommended_action"] = "Investigate missing-value-rate change"
        frame.loc[type_change, "recommended_action"] = "Review schema/type compatibility"
        frame = frame.sort_values(["drift_flag", "feature"], ascending=[False, True]).reset_index(drop=True)

    return {
        "features": frame,
        "schema": {
            "reference_only_columns": [c for c in ref_cols if c not in inc_set],
            "incoming_only_columns": [c for c in inc_cols if c not in ref_set],
            "shared_columns": [c for c in ref_cols if c in inc_set],
            "type_changed_columns": [c for c in ref_cols if c in inc_set and str(reference[c].dtype) != str(incoming[c].dtype)],
        },
        "reference_rows": int(len(reference)), "incoming_rows": int(len(incoming)),
        "reference_columns": len(ref_cols), "incoming_columns": len(inc_cols),
        "reference_duplicates": int(reference.duplicated().sum()),
        "incoming_duplicates": int(incoming.duplicated().sum()),
        "reference_duplicate_rate_pct": float(reference.duplicated().mean() * 100),
        "incoming_duplicate_rate_pct": float(incoming.duplicated().mean() * 100),
        "reference_missing_cells": int(reference.isna().sum().sum()),
        "incoming_missing_cells": int(incoming.isna().sum().sum()),
    }


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
        if not performance_verified:
            status = "INSUFFICIENT_EVIDENCE"
            recommendation = (
                "Collect labelled incoming observations before concluding that the model remains reliable."
            )
            reason = (
                "No monitored feature crossed the drift threshold, but incoming prediction performance "
                "has not been verified with enough labelled observations."
            )
        else:
            status = "CONTINUE"
            recommendation = (
                "Continue using the current model and monitor future data."
            )
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
