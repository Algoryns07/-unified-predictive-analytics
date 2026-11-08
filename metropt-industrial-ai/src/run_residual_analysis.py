from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
DOCS = ROOT / "docs"

DATA = OUTPUT / "metropt_minute_level_data.csv"

TARGETS = ["Oil_temperature", "TP3", "Motor_current"]

SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Flowmeter", "Motor_current"
]

LAGS = [1, 2, 5, 10, 15]
SEEDS = [17, 42, 73]

FOLDS = [
    {"name": "Window 1", "train_end": 0.55, "test_end": 0.70},
    {"name": "Window 2", "train_end": 0.70, "test_end": 0.85},
    {"name": "Window 3", "train_end": 0.85, "test_end": 1.00},
]

MODEL_PARAMETERS = {
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "learning_rate": 0.08,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0
}


def calculate_metrics(actual, predicted):
    residual = actual - predicted

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "mean_residual": float(np.mean(residual)),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "P90_absolute_error": float(np.percentile(np.abs(residual), 90)),
        "P95_absolute_error": float(np.percentile(np.abs(residual), 95)),
        "P99_absolute_error": float(np.percentile(np.abs(residual), 99)),
    }


def prepare_target_data(data, target):
    frame = data.copy()

    gap = frame["timestamp"].diff().ne(pd.Timedelta(minutes=1))
    frame["_segment"] = gap.cumsum()

    groups = frame.groupby("_segment", sort=False)

    X = frame[SENSORS].copy()

    for lag in LAGS:
        X[f"{target}_lag_{lag}"] = groups[target].shift(lag)

    X[f"{target}_rolling_mean_5"] = groups[target].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )

    X[f"{target}_rolling_mean_15"] = groups[target].transform(
        lambda s: s.rolling(15, min_periods=15).mean()
    )

    y = groups[target].shift(-1)
    target_time = groups["timestamp"].shift(-1)

    valid = X.notna().all(axis=1) & y.notna() & target_time.notna()

    X = X.loc[valid].astype(float)
    y = y.loc[valid].astype(float)
    target_time = target_time.loc[valid]

    return X, y, target_time


def create_model(seed):
    return HistGradientBoostingRegressor(
        **MODEL_PARAMETERS,
        random_state=seed
    )


def assign_temporal_folds(target_time):
    ordered_times = target_time.sort_values().reset_index(drop=True)
    assignments = []

    for fold in FOLDS:
        start_position = int(len(ordered_times) * fold["train_end"])

        if fold["test_end"] < 1.0:
            end_position = int(len(ordered_times) * fold["test_end"])
            end_time = ordered_times.iloc[end_position]
        else:
            end_time = ordered_times.iloc[-1] + pd.Timedelta(minutes=1)

        start_time = ordered_times.iloc[start_position]

        assignments.append({
            "name": fold["name"],
            "start_time": start_time,
            "end_time": end_time,
            "train_mask": target_time < start_time,
            "test_mask": (
                (target_time >= start_time)
                & (target_time < end_time)
            )
        })

    return assignments


def calculate_contiguous_rolling_mae(frame, window=60):
    frame = frame.sort_values("timestamp").copy()

    gap = frame["timestamp"].diff().ne(pd.Timedelta(minutes=1))
    frame["_segment"] = gap.cumsum()

    frame["rolling_MAE_60min"] = (
        frame.groupby("_segment")["absolute_error"]
        .transform(
            lambda s: s.rolling(
                window=window,
                min_periods=15
            ).mean()
        )
    )

    return frame


def analyse_target(data, target):
    print(f"\n{'=' * 65}")
    print("TARGET:", target)
    print("=" * 65)

    X, y, target_time = prepare_target_data(data, target)
    fold_assignments = assign_temporal_folds(target_time)

    target_predictions = []
    summary_rows = []

    for fold in fold_assignments:
        X_train = X.loc[fold["train_mask"]]
        y_train = y.loc[fold["train_mask"]]

        X_test = X.loc[fold["test_mask"]]
        y_test = y.loc[fold["test_mask"]]
        time_test = target_time.loc[fold["test_mask"]]

        if len(X_train) == 0 or len(X_test) == 0:
            raise RuntimeError(
                f"Empty split for {target}, {fold['name']}"
            )

        print(
            f"\n{fold['name']}: "
            f"Train={len(X_train):,}, Test={len(X_test):,}"
        )

        for seed in SEEDS:
            print(f"Training HistGradientBoosting, seed={seed}...")

            model = create_model(seed)
            model.fit(X_train, y_train)

            predicted = model.predict(X_test)
            actual = y_test.to_numpy()

            residual = actual - predicted

            prediction_frame = pd.DataFrame({
                "timestamp": time_test.to_numpy(),
                "target": target,
                "fold": fold["name"],
                "seed": seed,
                "actual": actual,
                "predicted": predicted,
                "residual": residual,
                "absolute_error": np.abs(residual)
            })

            target_predictions.append(prediction_frame)

            metrics = calculate_metrics(actual, predicted)

            summary_rows.append({
                "target": target,
                "fold": fold["name"],
                "seed": seed,
                "training_observations": len(X_train),
                "testing_observations": len(X_test),
                **metrics
            })

            print(
                f"  MAE={metrics['MAE']:.5f} | "
                f"RMSE={metrics['RMSE']:.5f} | "
                f"R2={metrics['R2']:.5f} | "
                f"Bias={metrics['mean_residual']:.5f}"
            )

    return (
        pd.concat(target_predictions, ignore_index=True),
        summary_rows
    )


def generate_plots(predictions, target, plot_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable. Skipping plots.")
        return False

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Seed 42 is used as a representative visualization.
    # All three seeds remain available in the prediction CSV.
    frame = predictions[
        (predictions["target"] == target)
        & (predictions["seed"] == 42)
    ].copy()

    frame = frame.sort_values("timestamp")

    fig, axes = plt.subplots(3, 1, figsize=(15, 13))

    # 1. Actual versus predicted values.
    axes[0].plot(
        frame["timestamp"],
        frame["actual"],
        label="Actual",
        linewidth=0.8
    )

    axes[0].plot(
        frame["timestamp"],
        frame["predicted"],
        label="Predicted",
        linewidth=0.8
    )

    axes[0].set_title(f"{target}: Actual vs Predicted")
    axes[0].set_ylabel("Sensor value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Residual distribution.
    axes[1].hist(
        frame["residual"],
        bins=60,
        edgecolor="black"
    )

    axes[1].axvline(0, linestyle="--")
    axes[1].set_title(f"{target}: Residual Distribution")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    # 3. Rolling MAE, reset at recording gaps.
    frame = calculate_contiguous_rolling_mae(frame)

    axes[2].plot(
        frame["timestamp"],
        frame["rolling_MAE_60min"],
        linewidth=0.8
    )

    axes[2].set_title(f"{target}: Rolling 60-minute MAE")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("Rolling MAE")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    plot_path = plot_dir / f"{target.lower()}_residual_analysis.png"

    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Plot generated:", plot_path)

    return True


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    print("Loading cached MetroPT minute-level dataset...")
    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not chronological.")

    if data[SENSORS].isna().any().any():
        raise ValueError("Missing sensor values detected.")

    all_predictions = []
    all_summaries = []

    for target in TARGETS:
        predictions, summaries = analyse_target(data, target)

        all_predictions.append(predictions)
        all_summaries.extend(summaries)

    prediction_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    prediction_path = OUTPUT / "metropt_residual_predictions.csv"
    summary_path = OUTPUT / "metropt_residual_summary.csv"
    worst_path = OUTPUT / "metropt_worst_predictions.csv"
    report_path = DOCS / "METROPT_RESIDUAL_ANALYSIS_REPORT.md"

    prediction_df.to_csv(prediction_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Worst 20 predictions for each target, fold and seed.
    worst = (
        prediction_df.sort_values(
            "absolute_error",
            ascending=False
        )
        .groupby(["target", "fold", "seed"], sort=False)
        .head(20)
        .sort_values(
            ["target", "fold", "seed", "absolute_error"],
            ascending=[True, True, True, False]
        )
    )

    worst.to_csv(worst_path, index=False)

    # Aggregation across folds and seeds.
    aggregate = (
        summary_df.groupby("target")
        .agg(
            experiments=("RMSE", "count"),
            mean_MAE=("MAE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_R2=("R2", "mean"),
            mean_bias=("mean_residual", "mean"),
            mean_P95_absolute_error=("P95_absolute_error", "mean"),
            mean_P99_absolute_error=("P99_absolute_error", "mean")
        )
        .reset_index()
    )

    aggregate_path = OUTPUT / "metropt_residual_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    plot_dir = OUTPUT / "residual_plots"
    plots_generated = {}

    for target in TARGETS:
        plots_generated[target] = generate_plots(
            prediction_df,
            target,
            plot_dir
        )

    report = [
        "# MetroPT Residual and Prediction Error Analysis",
        "",
        "## Experimental design",
        "",
        "- HistGradientBoosting with the Phase 2 hyperparameters.",
        "- Three chronological test windows.",
        "- Three seeds: 17, 42 and 73.",
        "- Targets: Oil_temperature, TP3 and Motor_current.",
        "- Residual = actual minus predicted.",
        "- Rolling MAE uses a 60-observation window.",
        "- Rolling calculations reset at timestamp gaps.",
        "- Seed 42 is used for representative visualizations.",
        "",
        "## Aggregate residual metrics",
        "",
        "| Target | Mean MAE | Mean RMSE | RMSE Std | Mean R² | Mean Bias | Mean P95 Absolute Error | Mean P99 Absolute Error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]

    for row in aggregate.itertuples(index=False):
        report.append(
            f"| {row.target} | {row.mean_MAE:.6f} | "
            f"{row.mean_RMSE:.6f} | {row.std_RMSE:.6f} | "
            f"{row.mean_R2:.6f} | {row.mean_bias:.6f} | "
            f"{row.mean_P95_absolute_error:.6f} | "
            f"{row.mean_P99_absolute_error:.6f} |"
        )

    report.extend([
        "",
        "## Per-window and seed results",
        "",
        "| Target | Window | Seed | MAE | RMSE | R² | Mean Bias | P95 Absolute Error | P99 Absolute Error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    ])

    for row in summary_df.itertuples(index=False):
        report.append(
            f"| {row.target} | {row.fold} | {row.seed} | "
            f"{row.MAE:.6f} | {row.RMSE:.6f} | "
            f"{row.R2:.6f} | {row.mean_residual:.6f} | "
            f"{row.P95_absolute_error:.6f} | "
            f"{row.P99_absolute_error:.6f} |"
        )

    report.extend([
        "",
        "## Interpretation guidance",
        "",
        "- Positive mean residual indicates average underprediction.",
        "- Negative mean residual indicates average overprediction.",
        "- Large P95 or P99 absolute errors indicate a heavier error tail.",
        "- Increasing rolling MAE suggests a period of deteriorating prediction accuracy.",
        "- A residual distribution centered near zero does not by itself establish calibrated uncertainty.",
        "- Associations between errors and operating periods are descriptive, not causal.",
        "",
        "## Limitations",
        "",
        "- This analysis uses the same chronological test windows as Phase 3.",
        "- HistGradientBoosting was selected using the earlier validation experiment.",
        "- Seed 42 plots are representative visualizations, not a separate evaluation.",
        "- High-error timestamps require further investigation before being labelled anomalies.",
        "- Prediction uncertainty and retraining effectiveness have not yet been established.",
        "",
        "## Generated visualizations",
        ""
    ])

    for target, generated in plots_generated.items():
        report.append(
            f"- {target}: "
            + (
                f"`outputs/residual_plots/{target.lower()}_residual_analysis.png`"
                if generated else "Plot unavailable."
            )
        )

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n========== AGGREGATE RESIDUAL RESULTS ==========")
    print(aggregate.to_string(index=False))

    print("\n========== GENERATED FILES ==========")
    for path in [
        prediction_path,
        summary_path,
        aggregate_path,
        worst_path,
        report_path
    ]:
        print(path)

    print("\nResidual analysis completed.")


if __name__ == "__main__":
    main()
