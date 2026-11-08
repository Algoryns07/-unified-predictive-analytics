import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_detection import detect_feature_drift

from performance_monitoring import (
    calculate_error_metrics,
    evaluate_performance_deterioration,
    identify_sustained_deterioration,
    calculate_signal_association,
)

DATA_PATH = ROOT / "outputs" / "metropt_minute_level_data.csv"
OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = ROOT / "docs" / "METROPT_PERFORMANCE_AWARE_DRIFT_REPORT.md"

TARGETS = ["Oil_temperature", "TP3", "Motor_current"]

SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Flowmeter", "Motor_current",
]

LAGS = [1, 2, 5, 10, 15]
ROLLING_WINDOWS = [5, 15]

WINDOW_SIZE = 500
THRESHOLD_MULTIPLIERS = [1.25, 1.5, 2.0]

PSI_THRESHOLD = 0.20
KS_Q_THRESHOLD = 0.05

RANDOM_STATE = 42


def prepare_target_data(data, target):

    frame = data.copy()

    gap = frame["timestamp"].diff().ne(
        pd.Timedelta(minutes=1)
    )

    frame["_segment"] = gap.cumsum()

    groups = frame.groupby("_segment", sort=False)

    X = frame[SENSORS].copy()

    for lag in LAGS:
        X[f"{target}_lag_{lag}"] = groups[target].shift(lag)

    for window in ROLLING_WINDOWS:

        X[f"{target}_rolling_mean_{window}"] = (
            groups[target].transform(
                lambda series: series.rolling(
                    window,
                    min_periods=window,
                ).mean()
            )
        )

    y = groups[target].shift(-1)
    target_time = groups["timestamp"].shift(-1)

    valid = (
        X.notna().all(axis=1)
        & y.notna()
        & target_time.notna()
    )

    result = X.loc[valid].astype(float).copy()

    result["_actual"] = y.loc[valid].to_numpy()
    result["_target_time"] = target_time.loc[valid].to_numpy()

    result = result.reset_index(drop=True)

    result["_position"] = np.arange(len(result))

    return result


def get_contiguous_blocks(frame):

    times = pd.to_datetime(frame["_target_time"])

    breaks = times.diff().ne(pd.Timedelta(minutes=1))
    breaks.iloc[0] = True

    block_ids = breaks.cumsum()

    blocks = []

    for block_id, indices in frame.groupby(
        block_ids,
        sort=False,
    ).groups.items():

        block = frame.loc[indices].reset_index(drop=True)

        blocks.append((int(block_id), block))

    return blocks


def evaluate_drift(reference, current):

    features = [
        column
        for column in current.columns
        if column not in [
            "_actual",
            "_target_time",
            "_position",
        ]
    ]

    result = detect_feature_drift(
        reference[features],
        current[features],
        psi_threshold=PSI_THRESHOLD,
        q_value_threshold=KS_Q_THRESHOLD,
        bins=10,
        min_drifted_features=2,
    )

    table = result["feature_results"]

    return {
        "drift_alert": bool(result["drift_detected"]),
        "drifted_features": int(table["feature_drift"].sum()),
        "psi_drifted_features": int(table["psi_drift"].sum()),
        "ks_drifted_features": int(table["ks_drift"].sum()),
        "median_psi": float(table["psi"].median()),
        "maximum_psi": float(table["psi"].max()),
    }


def main():

    print("Loading MetroPT minute-level data...")

    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    window_rows = []
    threshold_rows = []
    summary_rows = []

    for target in TARGETS:

        print("\n" + "=" * 75)
        print("TARGET:", target)
        print("=" * 75)

        prepared = prepare_target_data(data, target)

        n = len(prepared)

        train_end = int(n * 0.55)
        validation_end = int(n * 0.70)

        train = prepared.iloc[:train_end].copy()

        validation = prepared.iloc[
            train_end:validation_end
        ].copy()

        test_count = n - validation_end

        feature_columns = [
            column for column in prepared.columns
            if column not in [
                "_actual",
                "_target_time",
                "_position",
            ]
        ]

        X_train = train[feature_columns]
        y_train = train["_actual"]

        print("Prepared observations:", f"{n:,}")
        print("Training observations:", f"{len(train):,}")
        print("Validation observations:", f"{len(validation):,}")
        print("Untouched test observations:", f"{test_count:,}")

        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.08,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

        print("Training initial model...")

        model.fit(X_train, y_train)

        validation_predictions = model.predict(
            validation[feature_columns]
        )

        validation = validation.copy()

        validation["_prediction"] = validation_predictions

        blocks = get_contiguous_blocks(validation)

        monitoring_rows = []

        for block_id, block in blocks:

            number_of_windows = len(block) // WINDOW_SIZE

            for index in range(number_of_windows):

                start = index * WINDOW_SIZE
                end = start + WINDOW_SIZE

                window = block.iloc[start:end].copy()

                metrics = calculate_error_metrics(
                    window["_actual"],
                    window["_prediction"],
                )

                drift = evaluate_drift(
                    train[feature_columns],
                    window[feature_columns],
                )

                timestamps = pd.to_datetime(
                    window["_target_time"]
                )

                monitoring_rows.append({

                    "target": target,
                    "block_id": block_id,
                    "window_number": len(monitoring_rows) + 1,
                    "window_start": timestamps.iloc[0],
                    "window_end": timestamps.iloc[-1],
                    "observations": len(window),

                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "bias": metrics["bias"],

                    "drift_alert": drift["drift_alert"],
                    "drifted_features": drift["drifted_features"],
                    "psi_drifted_features": drift["psi_drifted_features"],
                    "ks_drifted_features": drift["ks_drifted_features"],
                    "median_psi": drift["median_psi"],
                    "maximum_psi": drift["maximum_psi"],
                })

        if len(monitoring_rows) < 4:
            raise RuntimeError(
                f"Insufficient monitoring windows for {target}."
            )

        windows = pd.DataFrame(monitoring_rows)

        # First half of validation windows establishes baseline.
        # Second half is used for threshold evaluation.

        baseline_count = len(windows) // 2

        baseline_windows = windows.iloc[:baseline_count]
        evaluation_windows = windows.iloc[baseline_count:].copy()

        baseline_mae = float(
            baseline_windows["mae"].mean()
        )

        baseline_rmse = float(
            baseline_windows["rmse"].mean()
        )

        if baseline_mae <= 0:
            raise RuntimeError(
                f"Invalid MAE baseline for {target}."
            )

        windows["baseline_period"] = (
            windows.index < baseline_count
        )

        windows["baseline_mae"] = baseline_mae
        windows["baseline_rmse"] = baseline_rmse

        windows["error_ratio"] = (
            windows["mae"] / baseline_mae
        )

        # Candidate thresholds are evaluated only on
        # the second half of validation windows.

        for multiplier in THRESHOLD_MULTIPLIERS:

            evaluation_windows = evaluation_windows.copy()

            evaluation_windows["performance_alert"] = (
                evaluation_windows["mae"]
                > baseline_mae * multiplier
            )

            evaluation_windows["sustained_alert"] = (
                identify_sustained_deterioration(
                    evaluation_windows["performance_alert"],
                    evaluation_windows["block_id"],
                    required_windows=2,
                )
            )

            association = calculate_signal_association(
                evaluation_windows["drift_alert"],
                evaluation_windows["performance_alert"],
            )

            sustained_association = calculate_signal_association(
                evaluation_windows["drift_alert"],
                evaluation_windows["sustained_alert"],
            )

            threshold_rows.append({

                "target": target,
                "threshold_multiplier": multiplier,

                "baseline_windows": baseline_count,

                "evaluation_windows": len(evaluation_windows),

                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,

                "performance_alerts": int(
                    evaluation_windows["performance_alert"].sum()
                ),

                "sustained_performance_alerts": int(
                    evaluation_windows["sustained_alert"].sum()
                ),

                "drift_alerts": int(
                    evaluation_windows["drift_alert"].sum()
                ),

                "drift_performance_cooccurrences": association["both"],

                "drift_sustained_cooccurrences": sustained_association["both"],

                "performance_alert_rate_pct": (
                    association["performance_rate_pct"]
                ),

                "drift_alert_rate_pct": (
                    association["drift_rate_pct"]
                ),

                "drift_performance_cooccurrence_pct": (
                    association["cooccurrence_rate_pct"]
                ),
            })

        # Use 1.5x as the primary candidate for reporting.
        # This is not yet a finalized retraining threshold.

        primary = evaluation_windows.copy()

        # Preserve the normalized error measure in the primary
        # evaluation table used by the summary report.
        primary["error_ratio"] = (
            primary["mae"] / baseline_mae
        )

        primary["performance_alert"] = (
            primary["mae"] > baseline_mae * 1.5
        )

        primary["sustained_alert"] = (
            identify_sustained_deterioration(
                primary["performance_alert"],
                primary["block_id"],
                required_windows=2,
            )
        )

        windows["evaluation_period"] = (
            windows.index >= baseline_count
        )

        windows["performance_alert_1_5x"] = False
        windows["sustained_alert_1_5x"] = False

        windows.loc[
            primary.index,
            "performance_alert_1_5x"
        ] = primary["performance_alert"].to_numpy()

        windows.loc[
            primary.index,
            "sustained_alert_1_5x"
        ] = primary["sustained_alert"].to_numpy()

        association = calculate_signal_association(
            primary["drift_alert"],
            primary["performance_alert"],
        )

        summary_rows.append({

            "target": target,

            "monitoring_windows": len(windows),

            "baseline_windows": baseline_count,

            "evaluation_windows": len(primary),

            "baseline_mae": baseline_mae,

            "baseline_rmse": baseline_rmse,

            "evaluation_mean_mae": float(
                primary["mae"].mean()
            ),

            "evaluation_mean_rmse": float(
                primary["rmse"].mean()
            ),

            "evaluation_mean_error_ratio": float(
                primary["error_ratio"].mean()
            ),

            "drift_alert_rate_pct": association["drift_rate_pct"],

            "performance_alert_rate_pct": (
                association["performance_rate_pct"]
            ),

            "drift_performance_cooccurrences": (
                association["both"]
            ),

            "sustained_performance_alerts": int(
                primary["sustained_alert"].sum()
            ),

            "overall_validation_r2": float(
                r2_score(
                    validation["_actual"],
                    validation["_prediction"],
                )
            ),
        })

        windows["target"] = target

        window_rows.append(windows)

        print("\nMonitoring windows:", len(windows))
        print("Baseline windows:", baseline_count)
        print("Evaluation windows:", len(primary))
        print("Baseline MAE:", round(baseline_mae, 6))
        print("Baseline RMSE:", round(baseline_rmse, 6))
        print(
            "Evaluation mean MAE:",
            round(primary["mae"].mean(), 6)
        )
        print(
            "Evaluation mean error ratio:",
            round(primary["error_ratio"].mean(), 4)
        )
        print(
            "Feature drift alert rate:",
            round(association["drift_rate_pct"], 2),
            "%"
        )
        print(
            "Performance alert rate at 1.5x:",
            round(association["performance_rate_pct"], 2),
            "%"
        )
        print(
            "Drift/performance co-occurrences:",
            association["both"]
        )

    windows_df = pd.concat(
        window_rows,
        ignore_index=True,
    )

    thresholds_df = pd.DataFrame(
        threshold_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    REPORT_PATH.parent.mkdir(exist_ok=True)

    windows_path = OUTPUT_DIR / "metropt_performance_drift_windows.csv"
    thresholds_path = OUTPUT_DIR / "metropt_performance_drift_thresholds.csv"
    summary_path = OUTPUT_DIR / "metropt_performance_drift_summary.csv"

    windows_df.to_csv(windows_path, index=False)
    thresholds_df.to_csv(thresholds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report = []

    report.append("# MetroPT Performance-Aware Drift Evaluation")
    report.append("")
    report.append("## Experimental design")
    report.append("")
    report.append("- Initial training: first 55% of prepared observations.")
    report.append("- Validation: next 15%.")
    report.append("- Untouched test: final 30%.")
    report.append("- First half of validation monitoring windows establishes the error baseline.")
    report.append("- Second half evaluates candidate deterioration thresholds.")
    report.append("- Monitoring window: 500 consecutive observations.")
    report.append("- Model: HistGradientBoostingRegressor.")
    report.append("- Targets: Oil_temperature, TP3 and Motor_current.")
    report.append("")
    report.append("## Target-level summary")
    report.append("")
    report.append(summary_df.round(5).to_string(index=False))
    report.append("")
    report.append("## Candidate threshold comparison")
    report.append("")
    report.append(thresholds_df.round(5).to_string(index=False))
    report.append("")
    report.append("## Interpretation and limitations")
    report.append("")
    report.append(
        "Feature drift and predictive performance deterioration are "
        "evaluated separately. Their co-occurrence is descriptive and "
        "does not establish that feature drift caused prediction error."
    )
    report.append("")
    report.append(
        "The first half of validation windows is used to estimate the "
        "baseline error. Candidate thresholds are evaluated on the "
        "remaining validation windows. The final 30% is not used."
    )
    report.append("")
    report.append(
        "Two consecutive deterioration windows are required for the "
        "sustained alert. Consecutiveness is reset at recording gaps."
    )
    report.append("")
    report.append(
        "The threshold multipliers are candidate operating settings, "
        "not universally optimal values. They must be selected using "
        "validation evidence and then frozen before final testing."
    )
    report.append("")
    report.append(
        "The PSI and KS detector compares each monitoring window with "
        "the initial training reference. KS tests remain sensitive to "
        "temporal dependence. These results do not provide ground-truth "
        "false-positive rates."
    )
    report.append("")
    report.append(
        "No retraining is performed in this experiment. It evaluates "
        "whether drift signals coincide with predictive deterioration."
    )
    report.append("")
    report.append(
        "The final test period remains untouched and is reserved for "
        "the subsequent policy comparison."
    )

    REPORT_PATH.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n========== PHASE 7.3 SUMMARY ==========")
    print(summary_df.round(4).to_string(index=False))

    print("\n========== THRESHOLD COMPARISON ==========")
    print(thresholds_df.round(4).to_string(index=False))

    print("\n========== GENERATED FILES ==========")
    print(windows_path)
    print(thresholds_path)
    print(summary_path)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
