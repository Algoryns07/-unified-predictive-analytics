import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from controlled_drift import generate_synthetic_stream

from drift_detection import detect_feature_drift

from performance_monitoring import (
    calculate_error_metrics,
    identify_sustained_deterioration,
)


OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = ROOT / "docs" / "CONTROLLED_DRIFT_EXPERIMENT_REPORT.md"

SEEDS = [17, 42, 73, 101, 2026]

SCENARIOS = [
    "no_drift",
    "covariate_drift",
    "concept_drift",
]

N_TRAIN = 5000
N_CALIBRATION = 1000
N_EVALUATION = 4000

WINDOW_SIZE = 200
INJECTION_INDEX = 2000

PERFORMANCE_MULTIPLIER = 1.5

PSI_THRESHOLD = 0.20
KS_Q_THRESHOLD = 0.05

FEATURES = [
    "feature_0",
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
]


def train_model(X, y, seed):

    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.08,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    )

    model.fit(X, y)

    return model


def calculate_calibration_baseline(model, seed):

    calibration = generate_synthetic_stream(
        N_CALIBRATION,
        "no_drift",
        N_CALIBRATION // 2,
        seed + 10000,
    )

    predictions = model.predict(calibration["X"])

    errors = []

    for start in range(
        0,
        N_CALIBRATION,
        WINDOW_SIZE,
    ):

        end = start + WINDOW_SIZE

        metrics = calculate_error_metrics(
            calibration["y"][start:end],
            predictions[start:end],
        )

        errors.append(metrics["mae"])

    baseline_mae = float(np.mean(errors))

    if not np.isfinite(baseline_mae) or baseline_mae <= 0:
        raise RuntimeError("Invalid calibration MAE.")

    return baseline_mae


def evaluate_feature_drift(reference, current):

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
        "drift_alert": bool(result["drift_detected"]),
        "drifted_features": int(
            table["feature_drift"].sum()
        ),
        "median_psi": float(
            table["psi"].median()
        ),
        "maximum_psi": float(
            table["psi"].max()
        ),
    }


def run_single_experiment(
    scenario,
    seed,
    model,
    reference,
    baseline_mae,
):

    stream = generate_synthetic_stream(
        N_EVALUATION,
        scenario,
        INJECTION_INDEX,
        seed + 20000,
    )

    X = stream["X"]
    y = stream["y"]

    predictions = model.predict(X)

    rows = []

    for window_number, start in enumerate(
        range(0, N_EVALUATION, WINDOW_SIZE)
    ):

        end = start + WINDOW_SIZE

        current_X = X.iloc[start:end]

        metrics = calculate_error_metrics(
            y[start:end],
            predictions[start:end],
        )

        drift = evaluate_feature_drift(
            reference,
            current_X,
        )

        performance_alert = (
            metrics["mae"]
            > baseline_mae * PERFORMANCE_MULTIPLIER
        )

        rows.append({

            "scenario": scenario,
            "seed": seed,
            "window_number": window_number,
            "start_observation": start,
            "end_observation": end - 1,

            "injected": bool(
                stream["injected"][start:end].any()
            ),

            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "bias": metrics["bias"],

            "baseline_mae": baseline_mae,
            "error_ratio": (
                metrics["mae"] / baseline_mae
            ),

            "drift_alert": drift["drift_alert"],
            "drifted_features": drift["drifted_features"],
            "median_psi": drift["median_psi"],
            "maximum_psi": drift["maximum_psi"],

            "performance_alert": bool(performance_alert),
        })

    windows = pd.DataFrame(rows)

    windows["sustained_performance_alert"] = (
        identify_sustained_deterioration(
            windows["performance_alert"],
            [0] * len(windows),
            required_windows=2,
        )
    )

    injection_window = (
        INJECTION_INDEX // WINDOW_SIZE
    )

    if scenario == "no_drift":

        before = windows.copy()
        after = windows.copy()

    else:

        before = windows.iloc[:injection_window].copy()
        after = windows.iloc[injection_window:].copy()

    def rate(series):
        return float(100 * series.mean()) if len(series) else 0.0

    if scenario == "no_drift":

        false_drift_rate = rate(
            windows["drift_alert"]
        )

        false_performance_rate = rate(
            windows["performance_alert"]
        )

        false_sustained_rate = rate(
            windows["sustained_performance_alert"]
        )

        feature_detection = False
        performance_detection = False

        drift_delay = np.nan
        performance_delay = np.nan

    else:

        false_drift_rate = rate(
            before["drift_alert"]
        )

        false_performance_rate = rate(
            before["performance_alert"]
        )

        false_sustained_rate = rate(
            before["sustained_performance_alert"]
        )

        feature_detection = bool(
            after["drift_alert"].any()
        )

        performance_detection = bool(
            after["sustained_performance_alert"].any()
        )

        drift_positions = after.index[
            after["drift_alert"]
        ].tolist()

        performance_positions = after.index[
            after["sustained_performance_alert"]
        ].tolist()

        drift_delay = (
            (drift_positions[0] - injection_window + 1)
            * WINDOW_SIZE
            if drift_positions else np.nan
        )

        performance_delay = (
            (performance_positions[0] - injection_window + 1)
            * WINDOW_SIZE
            if performance_positions else np.nan
        )

    result = {

        "scenario": scenario,
        "seed": seed,

        "baseline_mae": baseline_mae,

        "pre_injection_windows": len(before),
        "post_injection_windows": len(after),

        "pre_false_drift_rate_pct": false_drift_rate,
        "pre_false_performance_rate_pct": false_performance_rate,
        "pre_false_sustained_rate_pct": false_sustained_rate,

        "post_drift_alert_rate_pct": rate(
            after["drift_alert"]
        ),

        "post_performance_alert_rate_pct": rate(
            after["performance_alert"]
        ),

        "post_sustained_alert_rate_pct": rate(
            after["sustained_performance_alert"]
        ),

        "feature_drift_detected": feature_detection,
        "performance_drift_detected": performance_detection,

        "feature_detection_delay_observations": drift_delay,
        "performance_detection_delay_observations": performance_delay,

        "post_mean_mae": float(after["mae"].mean()),
        "post_mean_error_ratio": float(after["error_ratio"].mean()),

    }

    return windows, result


def summarize_results(seed_results):

    rows = []

    for scenario, group in seed_results.groupby(
        "scenario",
        sort=False,
    ):

        if scenario == "no_drift":

            rows.append({

                "scenario": scenario,
                "experiments": len(group),

                "mean_false_drift_rate_pct": group[
                    "pre_false_drift_rate_pct"
                ].mean(),

                "mean_false_performance_rate_pct": group[
                    "pre_false_performance_rate_pct"
                ].mean(),

                "mean_false_sustained_rate_pct": group[
                    "pre_false_sustained_rate_pct"
                ].mean(),

                "feature_detection_rate_pct": np.nan,
                "performance_detection_rate_pct": np.nan,

                "mean_post_drift_alert_rate_pct": np.nan,
                "mean_post_performance_alert_rate_pct": np.nan,

                "mean_feature_detection_delay": np.nan,
                "mean_performance_detection_delay": np.nan,

                "mean_post_error_ratio": group[
                    "post_mean_error_ratio"
                ].mean(),

            })

        else:

            rows.append({

                "scenario": scenario,
                "experiments": len(group),

                "mean_false_drift_rate_pct": group[
                    "pre_false_drift_rate_pct"
                ].mean(),

                "mean_false_performance_rate_pct": group[
                    "pre_false_performance_rate_pct"
                ].mean(),

                "mean_false_sustained_rate_pct": group[
                    "pre_false_sustained_rate_pct"
                ].mean(),

                "feature_detection_rate_pct": (
                    100 * group["feature_drift_detected"].mean()
                ),

                "performance_detection_rate_pct": (
                    100 * group["performance_drift_detected"].mean()
                ),

                "mean_post_drift_alert_rate_pct": group[
                    "post_drift_alert_rate_pct"
                ].mean(),

                "mean_post_performance_alert_rate_pct": group[
                    "post_performance_alert_rate_pct"
                ].mean(),

                "mean_feature_detection_delay": group[
                    "feature_detection_delay_observations"
                ].mean(),

                "mean_performance_detection_delay": group[
                    "performance_detection_delay_observations"
                ].mean(),

                "mean_post_error_ratio": group[
                    "post_mean_error_ratio"
                ].mean(),

            })

    return pd.DataFrame(rows)


def main():

    print("========== CONTROLLED DRIFT EXPERIMENT ==========")

    print("Training observations:", N_TRAIN)
    print("Calibration observations:", N_CALIBRATION)
    print("Evaluation observations:", N_EVALUATION)
    print("Monitoring window:", WINDOW_SIZE)
    print("Injection point:", INJECTION_INDEX)
    print("Seeds:", SEEDS)

    window_frames = []
    seed_rows = []

    for seed in SEEDS:

        print("\n" + "=" * 70)
        print("RANDOM SEED:", seed)
        print("=" * 70)

        training = generate_synthetic_stream(
            N_TRAIN,
            "no_drift",
            N_TRAIN // 2,
            seed,
        )

        model = train_model(
            training["X"],
            training["y"],
            seed,
        )

        reference = training["X"].copy()

        baseline_mae = calculate_calibration_baseline(
            model,
            seed,
        )

        print("Calibration baseline MAE:", round(baseline_mae, 6))

        for scenario in SCENARIOS:

            print("\nScenario:", scenario)

            windows, result = run_single_experiment(
                scenario,
                seed,
                model,
                reference,
                baseline_mae,
            )

            window_frames.append(windows)
            seed_rows.append(result)

            print(
                "Mean post-injection error ratio:",
                round(result["post_mean_error_ratio"], 4),
            )

            print(
                "Post-injection feature drift rate:",
                round(result["post_drift_alert_rate_pct"], 2),
                "%",
            )

            print(
                "Post-injection performance alert rate:",
                round(result["post_performance_alert_rate_pct"], 2),
                "%",
            )

            print(
                "Sustained performance detection:",
                result["performance_drift_detected"],
            )

    windows_df = pd.concat(
        window_frames,
        ignore_index=True,
    )

    seed_df = pd.DataFrame(seed_rows)

    summary_df = summarize_results(seed_df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_PATH.parent.mkdir(exist_ok=True)

    windows_path = OUTPUT_DIR / "controlled_drift_windows.csv"
    seed_path = OUTPUT_DIR / "controlled_drift_seed_results.csv"
    summary_path = OUTPUT_DIR / "controlled_drift_summary.csv"

    windows_df.to_csv(windows_path, index=False)
    seed_df.to_csv(seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report = []

    report.append("# Controlled Drift Experiment Report")
    report.append("")
    report.append("## Experimental design")
    report.append("")
    report.append(f"- Training observations: {N_TRAIN}")
    report.append(f"- Calibration observations: {N_CALIBRATION}")
    report.append(f"- Evaluation observations: {N_EVALUATION}")
    report.append(f"- Monitoring window: {WINDOW_SIZE}")
    report.append(f"- Drift injection point: {INJECTION_INDEX}")
    report.append(f"- Random seeds: {SEEDS}")
    report.append(f"- Performance threshold: {PERFORMANCE_MULTIPLIER}x calibration MAE")
    report.append(f"- PSI threshold: {PSI_THRESHOLD}")
    report.append(f"- KS BH-adjusted q threshold: {KS_Q_THRESHOLD}")
    report.append("- Sustained performance alert: two consecutive windows.")
    report.append("")
    report.append("## Scenario definitions")
    report.append("")
    report.append(
        "- No drift: feature distributions and target relationship remain unchanged."
    )
    report.append(
        "- Covariate drift: two feature distributions shift; the target relationship remains unchanged."
    )
    report.append(
        "- Concept drift: the target-generating relationship changes; feature distributions remain unchanged."
    )
    report.append("")
    report.append("## Aggregate results")
    report.append("")
    report.append(summary_df.round(4).to_string(index=False))
    report.append("")
    report.append("## Per-seed results")
    report.append("")
    report.append(seed_df.round(4).to_string(index=False))
    report.append("")
    report.append("## Interpretation and limitations")
    report.append("")
    report.append(
        "These are controlled synthetic experiments. They test whether "
        "monitoring methods respond to known constructed changes; they do "
        "not establish the frequency or nature of drift in real MetroPT operations."
    )
    report.append("")
    report.append(
        "Covariate drift changes input distributions while retaining the "
        "same target function. Concept drift changes the target function "
        "while preserving the input distribution."
    )
    report.append("")
    report.append(
        "The model remains fixed throughout each evaluation stream. "
        "No retraining is performed in this experiment."
    )
    report.append("")
    report.append(
        "The performance threshold is calibrated on a separate synthetic "
        "calibration stream. Detection delay is measured in observations "
        "from the known injection point to the first qualifying alert."
    )
    report.append("")
    report.append(
        "A feature-drift alert is not equivalent to harmful predictive drift. "
        "Likewise, a performance alert is not automatically proof of concept drift."
    )
    report.append("")
    report.append(
        "The five seeds provide a small reproducibility check, not a "
        "comprehensive statistical guarantee."
    )
    report.append("")

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n========== PHASE 7.4 AGGREGATE SUMMARY ==========")
    print(summary_df.round(4).to_string(index=False))

    print("\n========== PER-SEED RESULTS ==========")
    print(seed_df.round(4).to_string(index=False))

    print("\n========== GENERATED FILES ==========")
    print(windows_path)
    print(seed_path)
    print(summary_path)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
