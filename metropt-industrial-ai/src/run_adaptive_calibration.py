from pathlib import Path
from collections import deque
import math
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

TARGETS = [
    "Oil_temperature",
    "TP3",
    "Motor_current"
]

SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Flowmeter", "Motor_current"
]

LAGS = [1, 2, 5, 10, 15]
SEEDS = [17, 42, 73]

FOLDS = [
    {
        "name": "Window 1",
        "train_end": 0.45,
        "calibration_end": 0.55,
        "test_end": 0.70
    },
    {
        "name": "Window 2",
        "train_end": 0.60,
        "calibration_end": 0.70,
        "test_end": 0.85
    },
    {
        "name": "Window 3",
        "train_end": 0.75,
        "calibration_end": 0.85,
        "test_end": 1.00
    }
]

MODEL_PARAMETERS = {
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "learning_rate": 0.08,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0
}

ALPHAS = {
    90: 0.10,
    95: 0.05
}

ROLLING_WINDOWS = [2000, 5000, 10000]

# Recalculate rolling quantiles every 100 test observations.
UPDATE_FREQUENCY = 100

# Save every tenth test observation for seed 42.
SAMPLE_EVERY = 10


def prepare_target_data(data, target):
    frame = data.copy()

    # Prevent lag and rolling features from crossing recording gaps.
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




def initialize_rolling_buffers(initial_residuals, windows):
    """Create bounded rolling buffers from calibration-period residuals."""
    values = np.asarray(initial_residuals, dtype=float)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Initial residuals must be a non-empty vector.")

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Initial residuals must be finite and non-negative.")

    buffers = {}

    for window in windows:
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("Rolling window sizes must be positive integers.")

        if len(values) < window:
            raise ValueError(
                f"Insufficient calibration observations for window {window}."
            )

        buffers[window] = deque(
            values[-window:].tolist(),
            maxlen=window
        )

    return buffers


def refresh_rolling_quantiles(buffers, alphas):
    """Calculate rolling quantiles from residuals observed so far."""
    refreshed = {}

    for window, buffer in buffers.items():
        for level, alpha in alphas.items():
            refreshed[(f"Rolling_{window}", level)] = conformal_quantile(
                buffer,
                alpha
            )

    return refreshed


def prequential_interval_step(prediction, actual, quantiles, buffers, alphas):
    """
    Construct intervals before revealing the current residual to buffers.

    Returns intervals calculated from past information, then updates
    rolling buffers after the current actual observation is evaluated.
    """
    prediction = float(prediction)
    actual = float(actual)

    if not np.isfinite(prediction) or not np.isfinite(actual):
        raise ValueError("Prediction and actual value must be finite.")

    intervals = {}

    for method in ["Fixed"] + [
        f"Rolling_{window}" for window in buffers
    ]:
        for level in alphas:
            q = quantiles[(method, level)]

            if q < 0 or np.isnan(q):
                raise ValueError("Calibration quantile must be non-negative.")

            intervals[(method, level)] = (
                prediction - q,
                prediction + q
            )

    # Update only after all current-observation intervals have been formed.
    residual = abs(actual - prediction)

    for buffer in buffers.values():
        buffer.append(residual)

    return intervals


def make_model(seed):
    return HistGradientBoostingRegressor(
        **MODEL_PARAMETERS,
        random_state=seed
    )


def conformal_quantile(residuals, alpha):
    """
    Finite-sample split-conformal order statistic.

    k = ceil((n + 1) * (1 - alpha))
    """
    values = np.asarray(residuals, dtype=float)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Residual buffer must be a non-empty vector.")

    if not np.isfinite(values).all():
        raise ValueError("Residual buffer contains non-finite values.")

    if (values < 0).any():
        raise ValueError("Absolute residuals cannot be negative.")

    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("Alpha must be a finite numeric value.")

    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("Alpha must be a finite numeric value.") from None

    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha must be finite and strictly between zero and one.")

    n = len(values)
    k = math.ceil((n + 1) * (1 - alpha))

    if k > n:
        return float("inf")

    return float(np.partition(values, k - 1)[k - 1])


def make_split_masks(target_time, fold):
    ordered = target_time.sort_values().reset_index(drop=True)

    train_position = int(len(ordered) * fold["train_end"])
    calibration_position = int(len(ordered) * fold["calibration_end"])
    test_position = int(len(ordered) * fold["test_end"])

    train_start = ordered.iloc[train_position]
    calibration_start = ordered.iloc[calibration_position]

    if fold["test_end"] < 1.0:
        test_end = ordered.iloc[test_position]
    else:
        test_end = ordered.iloc[-1] + pd.Timedelta(minutes=1)

    return {
        "train": target_time < train_start,
        "calibration": (
            (target_time >= train_start)
            & (target_time < calibration_start)
        ),
        "test": (
            (target_time >= calibration_start)
            & (target_time < test_end)
        )
    }


def create_method_names():
    return (
        ["Fixed"]
        + [f"Rolling_{window}" for window in ROLLING_WINDOWS]
    )


def evaluate_experiment(data, target, fold, seed):
    print(
        f"\nTarget={target} | "
        f"{fold['name']} | Seed={seed}"
    )

    X, y, target_time = prepare_target_data(data, target)
    masks = make_split_masks(target_time, fold)

    X_train = X.loc[masks["train"]]
    y_train = y.loc[masks["train"]]

    X_cal = X.loc[masks["calibration"]]
    y_cal = y.loc[masks["calibration"]]

    X_test = X.loc[masks["test"]]
    y_test = y.loc[masks["test"]]
    time_test = target_time.loc[masks["test"]]

    if min(len(X_train), len(X_cal), len(X_test)) == 0:
        raise RuntimeError("An experiment split is empty.")

    print(
        f"Train={len(X_train):,} | "
        f"Calibration={len(X_cal):,} | "
        f"Test={len(X_test):,}"
    )

    # Train one fixed predictive model.
    model = make_model(seed)
    model.fit(X_train, y_train)

    # Initial calibration residuals.
    calibration_prediction = model.predict(X_cal)

    initial_residuals = np.abs(
        y_cal.to_numpy() - calibration_prediction
    )

    # Generate predictions for the chronological test period.
    test_prediction = model.predict(X_test)

    actual = y_test.to_numpy(dtype=float)
    timestamps = time_test.to_numpy()

    point_mae = float(mean_absolute_error(actual, test_prediction))
    point_rmse = float(
        np.sqrt(mean_squared_error(actual, test_prediction))
    )
    point_r2 = float(r2_score(actual, test_prediction))

    print(
        f"Point model: MAE={point_mae:.5f} | "
        f"RMSE={point_rmse:.5f} | R2={point_r2:.5f}"
    )

    methods = create_method_names()

    # Initial quantiles are calculated only from calibration data.
    quantiles = {}

    for level, alpha in ALPHAS.items():
        quantiles[("Fixed", level)] = conformal_quantile(
            initial_residuals,
            alpha
        )

    # Initialize bounded rolling buffers and their calibration quantiles.
    buffers = initialize_rolling_buffers(
        initial_residuals,
        ROLLING_WINDOWS
    )

    quantiles.update(
        refresh_rolling_quantiles(buffers, ALPHAS)
    )

    # Metric storage.
    trackers = {}

    for method in methods:
        for level in ALPHAS:
            trackers[(method, level)] = {
                "covered": [],
                "widths": [],
                "lower_misses": 0,
                "upper_misses": 0,
                "quantiles": [],
                "quantile_updates": 0
            }

    sample_rows = []

    # Online simulation:
    # predict -> evaluate interval -> observe actual -> update residual buffer.
    for i in range(len(actual)):

        # Refresh rolling quantiles using only previously observed residuals.
        if i > 0 and i % UPDATE_FREQUENCY == 0:

            quantiles.update(
                refresh_rolling_quantiles(buffers, ALPHAS)
            )

            for window in ROLLING_WINDOWS:
                for level in ALPHAS:
                    method = f"Rolling_{window}"
                    trackers[(method, level)]["quantile_updates"] += 1

        prediction = float(test_prediction[i])
        observed = float(actual[i])

        absolute_error = abs(observed - prediction)

        intervals = prequential_interval_step(
            prediction,
            observed,
            quantiles,
            buffers,
            ALPHAS
        )

        for method in methods:
            for level in ALPHAS:

                q = float(quantiles[(method, level)])
                lower, upper = intervals[(method, level)]

                covered = lower <= observed <= upper
                lower_miss = observed < lower
                upper_miss = observed > upper

                tracker = trackers[(method, level)]

                tracker["covered"].append(bool(covered))
                tracker["widths"].append(float(upper - lower))
                tracker["lower_misses"] += int(lower_miss)
                tracker["upper_misses"] += int(upper_miss)
                tracker["quantiles"].append(float(q))

                if seed == 42 and i % SAMPLE_EVERY == 0:
                    sample_rows.append({
                        "timestamp": timestamps[i],
                        "target": target,
                        "fold": fold["name"],
                        "seed": seed,
                        "method": method,
                        "interval_level": level,
                        "actual": observed,
                        "predicted": prediction,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "absolute_error": absolute_error,
                        "interval_width": upper - lower,
                        "covered": bool(covered),
                        "calibration_quantile": q
                    })


    # Construct metric rows.
    metric_rows = []

    for method in methods:
        for level, alpha in ALPHAS.items():

            tracker = trackers[(method, level)]

            coverage = float(np.mean(tracker["covered"]))
            mean_width = float(np.mean(tracker["widths"]))

            metric_rows.append({
                "target": target,
                "fold": fold["name"],
                "seed": seed,
                "method": method,
                "interval_level": level,
                "alpha": alpha,
                "training_observations": len(X_train),
                "calibration_observations": len(X_cal),
                "testing_observations": len(X_test),
                "MAE": point_mae,
                "RMSE": point_rmse,
                "R2": point_r2,
                "coverage": coverage,
                "coverage_gap": level / 100 - coverage,
                "mean_width": mean_width,
                "median_width": float(np.median(tracker["widths"])),
                "lower_miss_rate": (
                    tracker["lower_misses"] / len(actual)
                ),
                "upper_miss_rate": (
                    tracker["upper_misses"] / len(actual)
                ),
                "mean_calibration_quantile": float(
                    np.mean(tracker["quantiles"])
                ),
                "min_calibration_quantile": float(
                    np.min(tracker["quantiles"])
                ),
                "max_calibration_quantile": float(
                    np.max(tracker["quantiles"])
                ),
                "quantile_update_frequency": UPDATE_FREQUENCY,
                "quantile_updates": tracker["quantile_updates"],
                "rolling_window_size": (
                    0 if method == "Fixed"
                    else int(method.split("_")[1])
                )
            })

    return metric_rows, sample_rows


def create_report(metrics, aggregate, path):
    lines = [
        "# MetroPT Adaptive Calibration Assessment",
        "",
        "## Objective",
        "",
        "Compare fixed split-conformal calibration with rolling residual calibration",
        "while keeping the predictive model unchanged.",
        "",
        "## Experimental configuration",
        "",
        "- Targets: Oil_temperature, TP3 and Motor_current.",
        "- Model: HistGradientBoostingRegressor.",
        "- Seeds: 17, 42 and 73.",
        "- Three chronological train/calibration/test windows.",
        "- Nominal interval levels: 90% and 95%.",
        "- Calibration quantiles use the finite-sample order statistic.",
        "- Rolling windows: 2,000, 5,000 and 10,000 residuals.",
        "- Rolling quantiles are refreshed every 100 test observations.",
        "- Actual test residuals are appended only after the corresponding prediction.",
        "- The predictive model is not retrained during this experiment.",
        "",
        "## Aggregate results",
        "",
        "| Target | Method | Level | Experiments | Coverage | Coverage SD | Mean Width | Median Width | Lower Miss | Upper Miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]

    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.method} | {row.interval_level}% | "
            f"{row.experiments} | {row.mean_coverage:.4f} | "
            f"{row.std_coverage:.4f} | {row.mean_width:.6f} | "
            f"{row.mean_median_width:.6f} | "
            f"{row.mean_lower_miss_rate:.4f} | "
            f"{row.mean_upper_miss_rate:.4f} |"
        )

    lines.extend([
        "",
        "## Window-level results",
        "",
        "| Target | Window | Method | Level | Coverage | Mean Width | Coverage Gap | Mean Quantile | RMSE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"
    ])

    window_summary = (
        metrics.groupby(
            ["target", "fold", "method", "interval_level"],
            as_index=False
        )
        .agg(
            coverage=("coverage", "mean"),
            mean_width=("mean_width", "mean"),
            coverage_gap=("coverage_gap", "mean"),
            mean_quantile=("mean_calibration_quantile", "mean"),
            RMSE=("RMSE", "mean")
        )
    )

    for row in window_summary.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.fold} | {row.method} | "
            f"{row.interval_level}% | {row.coverage:.4f} | "
            f"{row.mean_width:.6f} | {row.coverage_gap:.4f} | "
            f"{row.mean_quantile:.6f} | {row.RMSE:.6f} |"
        )

    lines.extend([
        "",
        "## Method comparison",
        "",
        "The fixed method uses the original calibration residuals throughout testing.",
        "Rolling methods use the most recent observed residuals in a bounded buffer.",
        "Quantile recalculation occurs every 100 test observations.",
        "",
        "The model remains fixed so that changes in interval performance can be",
        "attributed to calibration updates rather than predictive model retraining.",
        "",
        "## Important limitations",
        "",
        "- This is a prequential online simulation, not a deployment experiment.",
        "- Rolling calibration uses observed outcomes after they become available.",
        "- The first prediction in each test period uses calibration-period residuals.",
        "- Updating every 100 observations means quantiles can be stale within each block.",
        "- Serial dependence and temporal distribution changes violate the usual",
        "  exchangeability assumptions behind standard conformal coverage guarantees.",
        "- Rolling calibration does not guarantee nominal coverage.",
        "- Repeated seeds do not establish independent machine generalization.",
        "- No model retraining is performed in this phase.",
        "",
        "## Interpretation guidance",
        "",
        "- Compare coverage against the nominal interval level.",
        "- Examine interval width alongside coverage; wider intervals are not automatically better.",
        "- Check whether rolling calibration improves coverage in difficult windows.",
        "- Check whether improvement requires an unacceptable increase in interval width.",
        "- Inspect lower and upper miss rates for asymmetric error patterns.",
        "- Consider recalibration overhead separately from model retraining cost.",
        "",
        "## Next stage",
        "",
        "Integrate drift and impact monitoring with cost-sensitive model retraining.",
        "Compare the additional benefit of model updates against adaptive calibration alone."
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    print("Loading cached MetroPT minute-level dataset...")

    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if data[SENSORS].isna().any().any():
        raise ValueError("Missing sensor values detected.")

    metric_rows = []
    sample_frames = []

    total_experiments = len(TARGETS) * len(FOLDS) * len(SEEDS)
    current = 0

    for target in TARGETS:
        for fold in FOLDS:
            for seed in SEEDS:
                current += 1

                print(
                    f"\n========== EXPERIMENT "
                    f"{current}/{total_experiments} =========="
                )

                rows, samples = evaluate_experiment(
                    data,
                    target,
                    fold,
                    seed
                )

                metric_rows.extend(rows)

                if samples:
                    sample_frames.append(pd.DataFrame(samples))

    metrics = pd.DataFrame(metric_rows)

    aggregate = (
        metrics.groupby(
            ["target", "method", "interval_level"],
            as_index=False
        )
        .agg(
            experiments=("coverage", "count"),
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_width=("mean_width", "mean"),
            mean_median_width=("median_width", "mean"),
            mean_lower_miss_rate=("lower_miss_rate", "mean"),
            mean_upper_miss_rate=("upper_miss_rate", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_quantile=("mean_calibration_quantile", "mean"),
            mean_quantile_updates=("quantile_updates", "mean")
        )
    )

    sample_df = pd.concat(sample_frames, ignore_index=True)

    metrics_path = OUTPUT / "metropt_adaptive_calibration_metrics.csv"
    aggregate_path = OUTPUT / "metropt_adaptive_calibration_aggregate.csv"
    sample_path = OUTPUT / "metropt_adaptive_calibration_prediction_sample.csv"
    report_path = DOCS / "METROPT_ADAPTIVE_CALIBRATION_REPORT.md"

    metrics.to_csv(metrics_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    create_report(metrics, aggregate, report_path)

    print("\n========== AGGREGATE RESULTS ==========")
    print(aggregate.round(5).to_string(index=False))

    print("\n========== OUTPUT FILES ==========")

    for path in [
        metrics_path,
        aggregate_path,
        sample_path,
        report_path
    ]:
        print(path)

    print("\n========== EXPERIMENT SUMMARY ==========")
    print("Targets:", len(TARGETS))
    print("Chronological windows:", len(FOLDS))
    print("Seeds:", len(SEEDS))
    print("Methods:", create_method_names())
    print("Interval levels:", list(ALPHAS.keys()))
    print("Predictive model fits:", total_experiments)
    print("Metric evaluations:", len(metrics))
    print("Sampled prediction rows:", len(sample_df))

    print("\nAdaptive calibration experiment completed.")


if __name__ == "__main__":
    main()
