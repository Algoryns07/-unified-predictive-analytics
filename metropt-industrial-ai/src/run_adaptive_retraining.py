from __future__ import annotations

import sys
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from drift_detection import detect_feature_drift


DATA_PATH = ROOT / "outputs" / "metropt_minute_level_data.csv"
OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = ROOT / "docs" / "METROPT_ADAPTIVE_RETRAINING_REPORT.md"

TARGETS = [
    "Oil_temperature",
    "TP3",
    "Motor_current",
]

SENSOR_FEATURES = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Flowmeter",
    "Motor_current",
]

LAGS = [1, 2, 5, 10, 15]
ROLLING_WINDOWS = [5, 15]

SEEDS = [17, 42, 73]

TRAIN_FRACTION = 0.55
VALIDATION_FRACTION = 0.15

MONITOR_WINDOW = 500
PERIODIC_INTERVAL = 5000
COOLDOWN_OBSERVATIONS = 2000

PERFORMANCE_MULTIPLIER = 1.5

PSI_THRESHOLD = 0.20
KS_Q_THRESHOLD = 0.05

IMPACT_MIN_IMPROVEMENT = 0.05
IMPACT_MIN_BENEFIT_COST_SCORE = 0.05

POLICIES = [
    "P1_No_Retraining",
    "P2_Periodic",
    "P3_Performance_Triggered",
    "P4_Impact_Aware",
]


def create_model(seed):
    return HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.08,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    )


def prepare_target_data(frame, target):
    """
    Predict the next consecutive one-minute target.

    Lag and rolling features are calculated independently
    inside uninterrupted one-minute segments.
    """

    # Preserve column order while removing the duplicate
    # target column already present in SENSOR_FEATURES.
    required = list(dict.fromkeys([
        "timestamp",
        target,
        *SENSOR_FEATURES,
    ]))

    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns for {target}: {sorted(missing)}"
        )

    data = frame[required].copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["timestamp"]
    ).sort_values(
        "timestamp"
    ).reset_index(drop=True)

    difference = data["timestamp"].diff()

    segment_id = (
        difference.ne(pd.Timedelta(minutes=1))
    ).cumsum()

    prepared_parts = []

    for _, block in data.groupby(
        segment_id,
        sort=False,
    ):

        block = block.copy()

        X = block[SENSOR_FEATURES].copy()

        for lag in LAGS:
            X[f"{target}_lag_{lag}"] = (
                block[target].shift(lag)
            )

        previous_target = block[target].shift(1)

        for window in ROLLING_WINDOWS:
            X[f"{target}_rolling_mean_{window}"] = (
                previous_target.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )

        y = block[target].shift(-1)

        prepared = X.copy()

        prepared["target"] = y
        prepared["timestamp"] = block["timestamp"]

        prepared = prepared.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if not prepared.empty:
            prepared_parts.append(prepared)

    if not prepared_parts:
        raise RuntimeError(
            f"No usable observations for {target}."
        )

    result = pd.concat(
        prepared_parts,
        ignore_index=True,
    )

    feature_columns = [
        column
        for column in result.columns
        if column not in {"target", "timestamp"}
    ]

    X = result[feature_columns].astype(float)
    y = result["target"].astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise RuntimeError(
            f"Non-finite features found for {target}."
        )

    if not np.isfinite(y.to_numpy()).all():
        raise RuntimeError(
            f"Non-finite targets found for {target}."
        )

    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        result["timestamp"].reset_index(drop=True),
    )


def calculate_metrics(actual, predicted):
    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
    }


def calculate_validation_baseline(
    model,
    X_validation,
    y_validation,
):
    """
    Estimate baseline MAE using chronological validation
    observations. The final test stream is never used.
    """

    predictions = model.predict(X_validation)

    window_errors = []

    for start in range(
        0,
        len(X_validation),
        MONITOR_WINDOW,
    ):

        end = min(
            start + MONITOR_WINDOW,
            len(X_validation),
        )

        if end - start < 100:
            continue

        error = mean_absolute_error(
            y_validation.iloc[start:end],
            predictions[start:end],
        )

        window_errors.append(float(error))

    if len(window_errors) < 2:
        raise RuntimeError(
            "Insufficient validation windows."
        )

    baseline = float(
        np.median(window_errors)
    )

    if not np.isfinite(baseline) or baseline <= 0:
        raise RuntimeError(
            "Invalid validation baseline."
        )

    return baseline, predictions


def evaluate_drift(reference, current):
    result = detect_feature_drift(
        reference,
        current,
        psi_threshold=PSI_THRESHOLD,
        q_value_threshold=KS_Q_THRESHOLD,
        bins=10,
        min_drifted_features=2,
    )

    table = result["feature_results"]

    return {
        "feature_drift_alert": bool(
            result["drift_detected"]
        ),
        "drifted_feature_count": int(
            table["feature_drift"].sum()
        ),
        "median_psi": float(
            table["psi"].median()
        ),
    }


def fit_model(X, y, seed):
    model = create_model(seed)

    start = time.perf_counter()

    model.fit(X, y)

    elapsed = time.perf_counter() - start

    return model, float(elapsed)


def evaluate_policy(
    policy,
    seed,
    target,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    reference,
    baseline_mae,
    initial_model,
    initial_fit_seconds,
):
    """
    Prequential policy evaluation.

    For every monitoring window:
      1. Predict using the current model.
      2. Reveal the actual targets.
      3. Evaluate prediction performance.
      4. Decide whether future predictions need a new model.
    """

    model = copy.deepcopy(initial_model)

    history_X = pd.concat(
        [X_train, X_validation],
        ignore_index=True,
    )

    history_y = pd.concat(
        [y_train, y_validation],
        ignore_index=True,
    )

    predictions_all = []
    actual_all = []

    window_rows = []
    retraining_rows = []

    alert_streak = 0
    observations_since_retraining = 0

    total_retraining_seconds = 0.0
    total_retraining_training_rows = 0

    candidate_attempts = 0
    candidate_acceptances = 0

    last_retraining_position = 0

    number_of_windows = int(
        np.ceil(len(X_test) / MONITOR_WINDOW)
    )

    for window_number, start in enumerate(
        range(0, len(X_test), MONITOR_WINDOW)
    ):

        end = min(
            start + MONITOR_WINDOW,
            len(X_test),
        )

        current_X = X_test.iloc[start:end]
        current_y = y_test.iloc[start:end]

        # ----------------------------------------------------
        # STEP 1: PREDICT BEFORE REVEALING ACTUAL TARGETS
        # ----------------------------------------------------

        current_predictions = model.predict(
            current_X
        )

        predictions_all.extend(
            current_predictions.tolist()
        )

        actual_all.extend(
            current_y.tolist()
        )

        metrics = calculate_metrics(
            current_y,
            current_predictions,
        )

        performance_alert = (
            metrics["mae"]
            > baseline_mae * PERFORMANCE_MULTIPLIER
        )

        if performance_alert:
            alert_streak += 1
        else:
            alert_streak = 0

        drift = evaluate_drift(
            reference,
            current_X,
        )

        observations_since_retraining += len(
            current_X
        )

        event = "none"
        candidate_improvement = np.nan
        candidate_cost_ratio = np.nan
        benefit_cost_score = np.nan

        # ----------------------------------------------------
        # STEP 2: DETERMINE RETRAINING DECISION
        # ----------------------------------------------------

        if policy == "P2_Periodic":

            should_retrain = (
                observations_since_retraining
                >= PERIODIC_INTERVAL
            )

            if should_retrain:
                event = "periodic_retraining"

        elif policy in {
            "P3_Performance_Triggered",
            "P4_Impact_Aware",
        }:

            should_retrain = (
                alert_streak >= 2
                and observations_since_retraining
                >= COOLDOWN_OBSERVATIONS
            )

            if should_retrain:
                event = "performance_trigger"

        else:
            should_retrain = False

        # ----------------------------------------------------
        # STEP 3: IMPACT-AWARE CHALLENGER EVALUATION
        # ----------------------------------------------------

        accepted_challenger = False

        if (
            policy == "P4_Impact_Aware"
            and should_retrain
        ):

            candidate_attempts += 1

            candidate, candidate_fit_seconds = fit_model(
                history_X,
                history_y,
                seed,
            )

            # Challenger fitting is a real computational cost,
            # even when the candidate is ultimately rejected.
            total_retraining_seconds += candidate_fit_seconds
            total_retraining_training_rows += len(history_X)

            candidate_predictions = candidate.predict(
                current_X
            )

            candidate_mae = mean_absolute_error(
                current_y,
                candidate_predictions,
            )

            relative_improvement = (
                metrics["mae"] - candidate_mae
            ) / max(metrics["mae"], 1e-12)

            candidate_cost_ratio = (
                candidate_fit_seconds
                / max(initial_fit_seconds, 1e-12)
            )

            benefit_cost_score = (
                relative_improvement
                / max(candidate_cost_ratio, 1e-12)
            )

            candidate_improvement = float(
                relative_improvement
            )

            accepted_challenger = (
                relative_improvement
                >= IMPACT_MIN_IMPROVEMENT
                and benefit_cost_score
                >= IMPACT_MIN_BENEFIT_COST_SCORE
            )

            if accepted_challenger:
                candidate_acceptances += 1
                event = "impact_approved"
            else:
                event = "impact_rejected"

        # ----------------------------------------------------
        # STEP 4: REVEAL CURRENT WINDOW AND UPDATE HISTORY
        # ----------------------------------------------------

        history_X = pd.concat(
            [history_X, current_X],
            ignore_index=True,
        )

        history_y = pd.concat(
            [history_y, current_y],
            ignore_index=True,
        )

        # ----------------------------------------------------
        # STEP 5: RETRAIN FOR FUTURE PREDICTIONS
        # ----------------------------------------------------

        if policy == "P4_Impact_Aware":

            if accepted_challenger:

                model, fit_seconds = fit_model(
                    history_X,
                    history_y,
                    seed,
                )

                total_retraining_seconds += (
                    fit_seconds
                )

                total_retraining_training_rows += (
                    len(history_X)
                )

                retraining_rows.append({
                    "target": target,
                    "seed": seed,
                    "policy": policy,
                    "window_number": window_number,
                    "training_rows": len(history_X),
                    "fit_seconds": fit_seconds,
                    "candidate_improvement_pct": (
                        100 * candidate_improvement
                    ),
                    "candidate_cost_ratio": (
                        candidate_cost_ratio
                    ),
                    "benefit_cost_score": (
                        benefit_cost_score
                    ),
                })

                observations_since_retraining = 0
                alert_streak = 0
                last_retraining_position = end

        elif should_retrain and policy in {
            "P2_Periodic",
            "P3_Performance_Triggered",
        }:

            model, fit_seconds = fit_model(
                history_X,
                history_y,
                seed,
            )

            total_retraining_seconds += fit_seconds

            total_retraining_training_rows += len(
                history_X
            )

            retraining_rows.append({
                "target": target,
                "seed": seed,
                "policy": policy,
                "window_number": window_number,
                "training_rows": len(history_X),
                "fit_seconds": fit_seconds,
                "candidate_improvement_pct": np.nan,
                "candidate_cost_ratio": np.nan,
                "benefit_cost_score": np.nan,
            })

            observations_since_retraining = 0
            alert_streak = 0
            last_retraining_position = end

        window_rows.append({
            "target": target,
            "seed": seed,
            "policy": policy,
            "window_number": window_number,
            "start_observation": start,
            "end_observation": end - 1,
            "observations": len(current_X),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "error_ratio": (
                metrics["mae"] / baseline_mae
            ),
            "performance_alert": bool(
                performance_alert
            ),
            "consecutive_alert_windows": alert_streak,
            "feature_drift_alert": (
                drift["feature_drift_alert"]
            ),
            "drifted_feature_count": (
                drift["drifted_feature_count"]
            ),
            "median_psi": drift["median_psi"],
            "decision": event,
            "candidate_improvement": (
                candidate_improvement
            ),
            "benefit_cost_score": (
                benefit_cost_score
            ),
        })

    # --------------------------------------------------------
    # FINAL TEST-STREAM METRICS
    # --------------------------------------------------------

    final_metrics = calculate_metrics(
        np.asarray(actual_all),
        np.asarray(predictions_all),
    )

    summary = {
        "target": target,
        "seed": seed,
        "policy": policy,
        "test_observations": len(actual_all),
        "baseline_validation_mae": baseline_mae,
        "test_mae": final_metrics["mae"],
        "test_rmse": final_metrics["rmse"],
        "test_r2": final_metrics["r2"],
        "total_absolute_error": float(
            np.sum(
                np.abs(
                    np.asarray(actual_all)
                    - np.asarray(predictions_all)
                )
            )
        ),
        "retraining_events": len(retraining_rows),
        "candidate_attempts": candidate_attempts,
        "candidate_acceptances": candidate_acceptances,
        "retraining_runtime_seconds": (
            total_retraining_seconds
        ),
        "retraining_training_rows": (
            total_retraining_training_rows
        ),
        "initial_fit_seconds": initial_fit_seconds,
    }

    return (
        summary,
        window_rows,
        retraining_rows,
    )


def main():

    print("========== PHASE 7.5 ==========")

    frame = pd.read_csv(DATA_PATH)

    print("Dataset rows:", len(frame))
    print("Targets:", TARGETS)
    print("Seeds:", SEEDS)
    print("Policies:", POLICIES)

    all_summaries = []
    all_windows = []
    all_retraining = []
    calibration_rows = []

    for target in TARGETS:

        print("\n" + "=" * 75)
        print("TARGET:", target)
        print("=" * 75)

        X, y, timestamps = prepare_target_data(
            frame,
            target,
        )

        n = len(X)

        train_end = int(
            n * TRAIN_FRACTION
        )

        validation_end = int(
            n * (
                TRAIN_FRACTION
                + VALIDATION_FRACTION
            )
        )

        X_train = X.iloc[:train_end].reset_index(drop=True)
        y_train = y.iloc[:train_end].reset_index(drop=True)

        X_validation = X.iloc[
            train_end:validation_end
        ].reset_index(drop=True)

        y_validation = y.iloc[
            train_end:validation_end
        ].reset_index(drop=True)

        X_test = X.iloc[
            validation_end:
        ].reset_index(drop=True)

        y_test = y.iloc[
            validation_end:
        ].reset_index(drop=True)

        print("Prepared observations:", n)
        print("Training:", len(X_train))
        print("Validation:", len(X_validation))
        print("Final test:", len(X_test))

        print(
            "Training end:",
            timestamps.iloc[train_end - 1],
        )

        print(
            "Validation end:",
            timestamps.iloc[validation_end - 1],
        )

        print(
            "Final test begins:",
            timestamps.iloc[validation_end],
        )

        for seed in SEEDS:

            print("\nSeed:", seed)

            initial_model, initial_fit_seconds = fit_model(
                X_train,
                y_train,
                seed,
            )

            baseline_mae, _ = calculate_validation_baseline(
                initial_model,
                X_validation,
                y_validation,
            )

            print(
                "Validation baseline MAE:",
                round(baseline_mae, 6),
            )

            calibration_rows.append({
                "target": target,
                "seed": seed,
                "validation_baseline_mae": baseline_mae,
                "initial_fit_seconds": initial_fit_seconds,
                "training_observations": len(X_train),
                "validation_observations": len(X_validation),
                "test_observations": len(X_test),
            })

            reference = X_train.iloc[
                :min(10000, len(X_train))
            ].copy()

            for policy in POLICIES:

                print("  Policy:", policy)

                summary, windows, retraining = (
                    evaluate_policy(
                        policy=policy,
                        seed=seed,
                        target=target,
                        X_train=X_train,
                        y_train=y_train,
                        X_validation=X_validation,
                        y_validation=y_validation,
                        X_test=X_test,
                        y_test=y_test,
                        reference=reference,
                        baseline_mae=baseline_mae,
                        initial_model=initial_model,
                        initial_fit_seconds=initial_fit_seconds,
                    )
                )

                all_summaries.append(summary)
                all_windows.extend(windows)
                all_retraining.extend(retraining)

                print(
                    "    MAE:",
                    round(summary["test_mae"], 6),
                    "| RMSE:",
                    round(summary["test_rmse"], 6),
                    "| R2:",
                    round(summary["test_r2"], 6),
                    "| Retrainings:",
                    summary["retraining_events"],
                    "| Runtime:",
                    round(
                        summary["retraining_runtime_seconds"],
                        3,
                    ),
                    "seconds",
                )

    summary_df = pd.DataFrame(all_summaries)
    windows_df = pd.DataFrame(all_windows)
    retraining_df = pd.DataFrame(all_retraining)
    calibration_df = pd.DataFrame(calibration_rows)

    aggregate = (
        summary_df.groupby(
            ["target", "policy"],
            sort=False,
        )
        .agg(
            mean_mae=("test_mae", "mean"),
            std_mae=("test_mae", "std"),
            mean_rmse=("test_rmse", "mean"),
            std_rmse=("test_rmse", "std"),
            mean_r2=("test_r2", "mean"),
            mean_retraining_events=(
                "retraining_events",
                "mean",
            ),
            mean_retraining_runtime_seconds=(
                "retraining_runtime_seconds",
                "mean",
            ),
            mean_retraining_training_rows=(
                "retraining_training_rows",
                "mean",
            ),
            mean_candidate_acceptances=(
                "candidate_acceptances",
                "mean",
            ),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_PATH.parent.mkdir(exist_ok=True)

    outputs = {
        "summary": OUTPUT_DIR / "adaptive_retraining_seed_results.csv",
        "aggregate": OUTPUT_DIR / "adaptive_retraining_aggregate.csv",
        "windows": OUTPUT_DIR / "adaptive_retraining_windows.csv",
        "events": OUTPUT_DIR / "adaptive_retraining_events.csv",
        "calibration": OUTPUT_DIR / "adaptive_retraining_calibration.csv",
    }

    summary_df.to_csv(outputs["summary"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    windows_df.to_csv(outputs["windows"], index=False)
    retraining_df.to_csv(outputs["events"], index=False)
    calibration_df.to_csv(outputs["calibration"], index=False)

    report = []

    report.append("# MetroPT Adaptive Retraining Policy Comparison")
    report.append("")
    report.append("## Experimental design")
    report.append("")
    report.append("- Dataset: MetroPT minute-level observations.")
    report.append("- Targets: Oil_temperature, TP3, Motor_current.")
    report.append("- Chronological split: 55% training, 15% validation, 30% final test.")
    report.append("- Random seeds: 17, 42, 73.")
    report.append("- Monitoring window: 500 observations.")
    report.append("- Periodic interval: 5,000 observations.")
    report.append("- Performance threshold: 1.5 times validation baseline MAE.")
    report.append("- Performance trigger: two consecutive alert windows.")
    report.append("- Cooldown: 2,000 observations.")
    report.append("- Impact approval: at least 5% observed improvement and benefit-cost score at least 0.05.")
    report.append("")
    report.append("## Policy definitions")
    report.append("")
    report.append("1. P1: fixed initial model.")
    report.append("2. P2: periodic retraining.")
    report.append("3. P3: sustained predictive-performance deterioration.")
    report.append("4. P4: challenger evaluation on the latest observed window, followed by impact/cost approval.")
    report.append("")
    report.append("## Aggregate results")
    report.append("")
    report.append(aggregate.round(5).to_string(index=False))
    report.append("")
    report.append("## Per-seed results")
    report.append("")
    report.append(summary_df.round(5).to_string(index=False))
    report.append("")
    report.append("## Methodological safeguards")
    report.append("")
    report.append("- Predictions are generated before the corresponding actual targets are used.")
    report.append("- Retraining decisions affect only future predictions.")
    report.append("- All policies begin with the same initial fitted model.")
    report.append("- Validation data calibrate the performance baseline and are not used to train the initial model.")
    report.append("- The final chronological test stream is not used to tune thresholds.")
    report.append("- Feature drift is recorded as diagnostic evidence and is not a mandatory retraining condition.")
    report.append("- Retraining cost is represented by measured fitting runtime and training-data volume, not invented monetary costs.")
    report.append("")
    report.append("## Interpretation limitations")
    report.append("")
    report.append("The impact-aware challenger is evaluated on the latest observed window, which is excluded from challenger fitting. After the decision, that window becomes available for subsequent training.")
    report.append("")
    report.append("The 5% improvement threshold and benefit-cost score are provisional operational choices, not statistically optimized universal thresholds.")
    report.append("")
    report.append("Results describe this dataset, feature construction, model and chronological evaluation period. They do not establish failure prediction or universal retraining superiority.")
    report.append("")
    report.append("Retraining policies are compared on the same test stream, but repeated sequential decisions mean the resulting observations are temporally dependent.")
    report.append("")

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n========== AGGREGATE RESULTS ==========")
    print(aggregate.round(5).to_string(index=False))

    print("\n========== OUTPUT FILES ==========")

    for name, path in outputs.items():
        print(name, ":", path)

    print("report:", REPORT_PATH)


if __name__ == "__main__":
    main()
