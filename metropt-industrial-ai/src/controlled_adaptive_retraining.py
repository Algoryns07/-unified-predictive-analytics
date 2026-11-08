from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from performance_monitoring import (
    calculate_error_metrics,
    evaluate_performance_deterioration,
    identify_sustained_deterioration,
)

from drift_detection import detect_feature_drift


WINDOW_SIZE = 200
PERIODIC_INTERVAL = 1000
COOLDOWN_OBSERVATIONS = 400

PERFORMANCE_MULTIPLIER = 1.5

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


def fit_model(X, y, seed):

    model = create_model(seed)

    start = time.perf_counter()

    model.fit(X, y)

    elapsed = time.perf_counter() - start

    return model, float(elapsed)


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

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


def calculate_calibration_baseline(
    model,
    X_calibration,
    y_calibration,
):

    predictions = model.predict(X_calibration)

    errors = []

    for start in range(
        0,
        len(X_calibration),
        WINDOW_SIZE,
    ):

        end = min(
            start + WINDOW_SIZE,
            len(X_calibration),
        )

        if end - start < 100:
            continue

        metrics = calculate_error_metrics(
            y_calibration[start:end],
            predictions[start:end],
        )

        errors.append(metrics["mae"])

    if len(errors) < 2:
        raise ValueError(
            "At least two calibration windows are required."
        )

    baseline = float(np.median(errors))

    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("Invalid calibration baseline.")

    return baseline


def evaluate_feature_drift(reference, current):

    result = detect_feature_drift(
        reference,
        current,
        psi_threshold=0.20,
        q_value_threshold=0.05,
        bins=10,
        min_drifted_features=2,
    )

    feature_results = result["feature_results"]

    if "psi" not in feature_results.columns:
        raise KeyError(
            "The drift detector's feature_results "
            "does not contain the expected 'psi' column."
        )

    median_psi = (
        float(feature_results["psi"].median())
        if not feature_results.empty
        else float("nan")
    )

    # Add reporting aliases while preserving the original
    # drift detector's complete result structure.
    result["feature_drift_alert"] = bool(
        result["drift_detected"]
    )

    result["median_psi"] = median_psi

    return result


def calculate_post_injection_retraining_delay(
    events,
    injection_index,
    window_size,
    scenario,
):
    """
    Return the delay to the first accepted retraining event
    at or after the injection window.

    Delay is measured from the injection observation to the
    end of the first qualifying monitoring window.
    """

    if scenario == "no_drift" or events.empty:
        return np.nan

    accepted = events[
        (events["event_type"] == "accepted_retraining")
        & (events["accepted"].astype(bool))
    ]

    injection_window = injection_index // window_size

    post_injection = accepted[
        accepted["window_number"] >= injection_window
    ]

    if post_injection.empty:
        return np.nan

    first_window = int(post_injection["window_number"].min())

    first_retraining_observation = (
        first_window + 1
    ) * window_size

    return float(
        max(0, first_retraining_observation - injection_index)
    )


def count_false_policy_trigger_events(
    events,
    policy,
    scenario,
    injection_window,
):
    """
    Count false policy-triggered retraining events.

    Periodic P2 updates are scheduled maintenance and are
    deliberately excluded from false-trigger counts.

    In no-drift experiments, every accepted P3/P4 policy
    trigger is counted as false.

    In drift experiments, accepted P3/P4 events before the
    injection window are counted as false triggers.
    """

    if events.empty or policy == "P2_Periodic":
        return 0

    accepted = events[
        (events["event_type"] == "accepted_retraining")
        & (events["accepted"].astype(bool))
    ]

    if scenario == "no_drift":
        return int(len(accepted))

    return int(
        (accepted["window_number"] < injection_window).sum()
    )


def evaluate_policy(
    policy,
    seed,
    scenario,
    X_train,
    y_train,
    X_calibration,
    y_calibration,
    X_evaluation,
    y_evaluation,
    baseline_mae,
    initial_model,
    initial_fit_seconds,
    injection_index,
):

    if policy not in POLICIES:
        raise ValueError("Unknown retraining policy.")

    model = copy.deepcopy(initial_model)

    # Calibration observations become available for
    # subsequent retraining after baseline estimation.
    history_X = pd.concat(
        [X_train, X_calibration],
        ignore_index=True,
    )

    history_y = np.concatenate([
        np.asarray(y_train, dtype=float),
        np.asarray(y_calibration, dtype=float),
    ])

    reference = X_train.copy()

    window_rows = []
    event_rows = []

    predictions_all = []
    actual_all = []

    alert_streak = 0
    observations_since_retraining = 0

    total_fit_seconds = 0.0
    total_training_rows = 0

    candidate_attempts = 0
    candidate_acceptances = 0

    number_of_windows = int(
        np.ceil(len(X_evaluation) / WINDOW_SIZE)
    )

    for window_number, start in enumerate(
        range(0, len(X_evaluation), WINDOW_SIZE)
    ):

        end = min(
            start + WINDOW_SIZE,
            len(X_evaluation),
        )

        current_X = X_evaluation.iloc[start:end]
        current_y = np.asarray(
            y_evaluation[start:end],
            dtype=float,
        )

        # -------------------------------------------------
        # 1. Predict before using current actual targets.
        # -------------------------------------------------

        current_predictions = model.predict(current_X)

        predictions_all.extend(
            current_predictions.tolist()
        )

        actual_all.extend(current_y.tolist())

        metrics = calculate_metrics(
            current_y,
            current_predictions,
        )

        performance_result = evaluate_performance_deterioration(
            metrics["mae"],
            baseline_mae,
            PERFORMANCE_MULTIPLIER,
        )

        performance_alert = performance_result["deteriorated"]

        if performance_alert:
            alert_streak += 1
        else:
            alert_streak = 0

        # Feature drift is diagnostic only.
        drift_result = evaluate_feature_drift(
            reference,
            current_X,
        )

        observations_since_retraining += len(current_X)

        sustained_alert = alert_streak >= 2

        should_retrain = False
        decision = "none"

        candidate_improvement = np.nan
        candidate_cost_ratio = np.nan
        benefit_cost_score = np.nan

        candidate_fit_seconds = 0.0
        candidate_training_rows = 0

        accepted = False

        # -------------------------------------------------
        # 2. Select policy decision.
        # -------------------------------------------------

        if policy == "P2_Periodic":

            should_retrain = (
                observations_since_retraining
                >= PERIODIC_INTERVAL
            )

            if should_retrain:
                decision = "periodic_retraining"

        elif policy in {
            "P3_Performance_Triggered",
            "P4_Impact_Aware",
        }:

            should_retrain = (
                sustained_alert
                and observations_since_retraining
                >= COOLDOWN_OBSERVATIONS
            )

            if should_retrain:
                decision = "performance_trigger"

        # -------------------------------------------------
        # 3. Evaluate impact-aware challenger.
        # -------------------------------------------------

        if policy == "P4_Impact_Aware" and should_retrain:

            candidate_attempts += 1

            candidate, candidate_fit_seconds = fit_model(
                history_X,
                history_y,
                seed,
            )

            candidate_training_rows = len(history_X)

            # Challenger fitting cost is counted even
            # if the challenger is subsequently rejected.
            total_fit_seconds += candidate_fit_seconds
            total_training_rows += candidate_training_rows

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

            accepted = (
                relative_improvement
                >= IMPACT_MIN_IMPROVEMENT
                and benefit_cost_score
                >= IMPACT_MIN_BENEFIT_COST_SCORE
            )

            if accepted:
                candidate_acceptances += 1
                decision = "impact_approved"
            else:
                decision = "impact_rejected"

            event_rows.append({
                "scenario": scenario,
                "seed": seed,
                "policy": policy,
                "window_number": window_number,
                "event_type": "candidate_evaluation",
                "accepted": bool(accepted),
                "training_rows": candidate_training_rows,
                "fit_seconds": candidate_fit_seconds,
                "candidate_improvement_pct": (
                    100 * candidate_improvement
                ),
                "candidate_cost_ratio": candidate_cost_ratio,
                "benefit_cost_score": benefit_cost_score,
            })

        # -------------------------------------------------
        # 4. Reveal current targets and update history.
        # -------------------------------------------------

        history_X = pd.concat(
            [history_X, current_X],
            ignore_index=True,
        )

        history_y = np.concatenate([
            history_y,
            current_y,
        ])

        # -------------------------------------------------
        # 5. Retrain only for future predictions.
        # -------------------------------------------------

        if policy == "P4_Impact_Aware":

            if accepted:

                model, fit_seconds = fit_model(
                    history_X,
                    history_y,
                    seed,
                )

                total_fit_seconds += fit_seconds
                total_training_rows += len(history_X)

                event_rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "policy": policy,
                    "window_number": window_number,
                    "event_type": "accepted_retraining",
                    "accepted": True,
                    "training_rows": len(history_X),
                    "fit_seconds": fit_seconds,
                    "candidate_improvement_pct": (
                        100 * candidate_improvement
                    ),
                    "candidate_cost_ratio": candidate_cost_ratio,
                    "benefit_cost_score": benefit_cost_score,
                })

                observations_since_retraining = 0
                alert_streak = 0

        elif should_retrain and policy in {
            "P2_Periodic",
            "P3_Performance_Triggered",
        }:

            model, fit_seconds = fit_model(
                history_X,
                history_y,
                seed,
            )

            total_fit_seconds += fit_seconds
            total_training_rows += len(history_X)

            event_rows.append({
                "scenario": scenario,
                "seed": seed,
                "policy": policy,
                "window_number": window_number,
                "event_type": "accepted_retraining",
                "accepted": True,
                "training_rows": len(history_X),
                "fit_seconds": fit_seconds,
                "candidate_improvement_pct": np.nan,
                "candidate_cost_ratio": np.nan,
                "benefit_cost_score": np.nan,
            })

            observations_since_retraining = 0
            alert_streak = 0

        # -------------------------------------------------
        # 6. Record monitoring window.
        # -------------------------------------------------

        window_rows.append({

            "scenario": scenario,
            "seed": seed,
            "policy": policy,

            "window_number": window_number,

            "start_observation": start,
            "end_observation": end - 1,

            "injected": bool(
                start >= injection_index
                if scenario != "no_drift"
                else False
            ),

            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],

            "error_ratio": (
                metrics["mae"] / baseline_mae
            ),

            "performance_alert": bool(
                performance_alert
            ),

            "sustained_performance_alert": bool(
                sustained_alert
            ),

            "feature_drift_alert": bool(
                drift_result["feature_drift_alert"]
            ),

            "drifted_feature_count": (
                drift_result["drifted_feature_count"]
            ),

            "median_psi": drift_result["median_psi"],

            "decision": decision,

            "candidate_improvement": (
                candidate_improvement
            ),

            "benefit_cost_score": (
                benefit_cost_score
            ),
        })

    # -----------------------------------------------------
    # 7. Final evaluation.
    # -----------------------------------------------------

    all_metrics = calculate_metrics(
        np.asarray(actual_all),
        np.asarray(predictions_all),
    )

    windows = pd.DataFrame(window_rows)
    events = pd.DataFrame(event_rows)

    injection_window = (
        injection_index // WINDOW_SIZE
    )

    pre_windows = windows[
        windows["end_observation"] < injection_index
    ]

    post_windows = windows[
        windows["start_observation"] >= injection_index
    ]

    if scenario == "no_drift":

        pre_windows = windows
        post_windows = windows.iloc[0:0]

    def average(frame, column):

        if len(frame) == 0:
            return np.nan

        return float(frame[column].mean())

    accepted_events = (
        events[
            events["event_type"] == "accepted_retraining"
        ]
        if len(events)
        else events
    )

    retraining_delay = calculate_post_injection_retraining_delay(
        events=events,
        injection_index=injection_index,
        window_size=WINDOW_SIZE,
        scenario=scenario,
    )

    if scenario != "no_drift":

        detected = post_windows[
            post_windows["sustained_performance_alert"]
        ]

        if len(detected):

            first_detection = int(
                detected.iloc[0]["start_observation"]
            )

            detection_delay = max(
                0,
                first_detection - injection_index
            )

        else:
            detection_delay = np.nan

    else:
        detection_delay = np.nan

    pre_injection_retraining = int(
        (
            accepted_events["window_number"]
            < injection_window
        ).sum()
    ) if len(accepted_events) else 0

    post_injection_retraining = int(
        (
            accepted_events["window_number"]
            >= injection_window
        ).sum()
    ) if len(accepted_events) else 0

    summary = {

        "scenario": scenario,
        "seed": seed,
        "policy": policy,

        "evaluation_observations": len(actual_all),

        "baseline_mae": baseline_mae,

        "overall_mae": all_metrics["mae"],
        "overall_rmse": all_metrics["rmse"],
        "overall_r2": all_metrics["r2"],

        "pre_drift_mae": average(
            pre_windows,
            "mae",
        ),

        "post_drift_mae": average(
            post_windows,
            "mae",
        ),

        "pre_drift_rmse": average(
            pre_windows,
            "rmse",
        ),

        "post_drift_rmse": average(
            post_windows,
            "rmse",
        ),

        "post_drift_error_ratio": average(
            post_windows,
            "error_ratio",
        ),

        "performance_detection_delay": detection_delay,

        "first_retraining_delay": retraining_delay,

        "pre_injection_retraining_events": (
            pre_injection_retraining
        ),

        "post_injection_retraining_events": (
            post_injection_retraining
        ),

        "total_retraining_events": len(
            accepted_events
        ),

        "candidate_attempts": candidate_attempts,
        "candidate_acceptances": candidate_acceptances,

        "candidate_acceptance_rate_pct": (
            100 * candidate_acceptances / candidate_attempts
            if candidate_attempts else 0.0
        ),

        "total_fit_seconds": total_fit_seconds,
        "total_training_rows": total_training_rows,

        "false_policy_trigger_events": (
            count_false_policy_trigger_events(
                events=events,
                policy=policy,
                scenario=scenario,
                injection_window=injection_window,
            )
        ),

        "false_performance_alerts": int(
            pre_windows["sustained_performance_alert"].sum()
        ),
    }

    return summary, window_rows, event_rows


def run_experiment(
    seeds,
    scenarios,
    n_train=5000,
    n_calibration=1000,
    n_evaluation=4000,
    injection_index=2000,
):

    from controlled_drift import generate_synthetic_stream

    summaries = []
    all_windows = []
    all_events = []

    for seed in seeds:

        print(
            f"\n========== SEED {seed} ==========",
            flush=True,
        )

        training = generate_synthetic_stream(
            n_train,
            "no_drift",
            n_train // 2,
            seed,
        )

        calibration = generate_synthetic_stream(
            n_calibration,
            "no_drift",
            n_calibration // 2,
            seed + 10000,
        )

        initial_model, initial_fit_seconds = fit_model(
            training["X"],
            training["y"],
            seed,
        )

        baseline_mae = calculate_calibration_baseline(
            initial_model,
            calibration["X"],
            calibration["y"],
        )

        print(
            f"Calibration MAE: {baseline_mae:.6f}",
            flush=True,
        )

        for scenario in scenarios:

            evaluation = generate_synthetic_stream(
                n_evaluation,
                scenario,
                injection_index,
                seed + 20000,
            )

            print(
                f"\nScenario: {scenario}",
                flush=True,
            )

            for policy in POLICIES:

                print(
                    f"  Running {policy}...",
                    flush=True,
                )

                summary, windows, events = evaluate_policy(
                    policy=policy,
                    seed=seed,
                    scenario=scenario,
                    X_train=training["X"],
                    y_train=training["y"],
                    X_calibration=calibration["X"],
                    y_calibration=calibration["y"],
                    X_evaluation=evaluation["X"],
                    y_evaluation=evaluation["y"],
                    baseline_mae=baseline_mae,
                    initial_model=initial_model,
                    initial_fit_seconds=initial_fit_seconds,
                    injection_index=injection_index,
                )

                summaries.append(summary)
                all_windows.extend(windows)
                all_events.extend(events)

                print(
                    f"    MAE={summary['overall_mae']:.5f} | "
                    f"Retrains={summary['total_retraining_events']} | "
                    f"Candidates={summary['candidate_attempts']}",
                    flush=True,
                )

    return (
        pd.DataFrame(summaries),
        pd.DataFrame(all_windows),
        pd.DataFrame(all_events),
    )
