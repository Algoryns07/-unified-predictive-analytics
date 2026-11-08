import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("metropt-industrial-ai")
sys.path.insert(0, str((ROOT / "src").resolve()))

from drift_detection import detect_feature_drift

DATA_PATH = ROOT / "outputs" / "metropt_minute_level_data.csv"
OUTPUT_DIR = ROOT / "outputs"

TARGETS = ["Oil_temperature", "TP3", "Motor_current"]

SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Flowmeter", "Motor_current",
]

LAGS = [1, 2, 5, 10, 15]
ROLLING_WINDOWS = [5, 15]

MONITORING_WINDOW = 500
REFERENCE_WINDOW = 500

PSI_THRESHOLD = 0.20
KS_Q_THRESHOLD = 0.05
MIN_DRIFTED_GROUPS = 2


def prepare_target_data(data, target):

    frame = data.copy()

    gap = frame["timestamp"].diff().ne(pd.Timedelta(minutes=1))
    frame["_segment"] = gap.cumsum()

    groups = frame.groupby("_segment", sort=False)
    X = frame[SENSORS].copy()

    for lag in LAGS:
        X[f"{target}_lag_{lag}"] = groups[target].shift(lag)

    for window in ROLLING_WINDOWS:
        X[f"{target}_rolling_mean_{window}"] = groups[target].transform(
            lambda series: series.rolling(
                window,
                min_periods=window
            ).mean()
        )

    y = groups[target].shift(-1)
    target_time = groups["timestamp"].shift(-1)

    valid = (
        X.notna().all(axis=1)
        & y.notna()
        & target_time.notna()
    )

    result = X.loc[valid].astype(float).copy()
    result["_target_time"] = target_time.loc[valid].to_numpy()

    result = result.reset_index(drop=True)
    result["_position"] = np.arange(len(result))

    return result


def create_sensor_groups(target):

    groups = {}

    for sensor in SENSORS:

        columns = [sensor]

        if sensor == target:

            columns.extend(
                f"{target}_lag_{lag}" for lag in LAGS
            )

            columns.extend(
                f"{target}_rolling_mean_{window}"
                for window in ROLLING_WINDOWS
            )

        groups[sensor] = columns

    return groups


def get_contiguous_blocks(frame):

    times = pd.to_datetime(frame["_target_time"])

    breaks = times.diff().ne(pd.Timedelta(minutes=1))
    breaks.iloc[0] = True

    block_ids = breaks.cumsum()

    return [
        frame.loc[indices].reset_index(drop=True)
        for _, indices in frame.groupby(
            block_ids,
            sort=False
        ).groups.items()
    ]


def evaluate_sensor_groups(reference, current, groups):

    group_results = []

    for group_name, columns in groups.items():

        result = detect_feature_drift(
            reference[columns],
            current[columns],
            psi_threshold=PSI_THRESHOLD,
            q_value_threshold=KS_Q_THRESHOLD,
            bins=10,
            min_drifted_features=1,
        )

        feature_table = result["feature_results"]

        group_results.append({

            "sensor_group": group_name,

            "group_alert": bool(
                result["drift_detected"]
            ),

            "flagged_features": int(
                feature_table["feature_drift"].sum()
            ),

            "psi_flagged_features": int(
                feature_table["psi_drift"].sum()
            ),

            "ks_flagged_features": int(
                feature_table["ks_drift"].sum()
            ),

            "maximum_psi": float(
                feature_table["psi"].max()
            ),

            # Correct column name: ks_q_value
            "minimum_q_value": float(
                feature_table["ks_q_value"].min()
            ),
        })

    return pd.DataFrame(group_results)


def get_recent_reference(
    prepared,
    start_position,
    current_start_time
):

    start = start_position - REFERENCE_WINDOW

    if start < 0:
        return None

    recent = prepared.iloc[
        start:start_position
    ].copy()

    if len(recent) != REFERENCE_WINDOW:
        return None

    times = pd.to_datetime(
        recent["_target_time"]
    )

    if not times.diff().iloc[1:].eq(
        pd.Timedelta(minutes=1)
    ).all():

        return None

    if (
        current_start_time - times.iloc[-1]
        != pd.Timedelta(minutes=1)
    ):

        return None

    return recent.drop(
        columns=["_target_time", "_position"]
    ).reset_index(drop=True)


def main():

    print("Loading MetroPT minute-level data...")

    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    all_window_results = []
    all_group_results = []

    for target in TARGETS:

        print("\n" + "=" * 75)
        print("TARGET:", target)
        print("=" * 75)

        prepared = prepare_target_data(
            data,
            target
        )

        n = len(prepared)

        train_end = int(n * 0.55)
        validation_end = int(n * 0.70)

        reference = prepared.iloc[
            :train_end
        ].drop(
            columns=["_target_time", "_position"]
        ).reset_index(drop=True)

        validation = prepared.iloc[
            train_end:validation_end
        ].copy()

        print("Prepared observations:", f"{n:,}")
        print("Initial reference:", f"{len(reference):,}")
        print("Validation:", f"{len(validation):,}")
        print("Untouched test:", f"{n-validation_end:,}")

        groups = create_sensor_groups(target)

        blocks = get_contiguous_blocks(validation)

        window_number = 0

        for block in blocks:

            number_of_windows = (
                len(block) // MONITORING_WINDOW
            )

            for index in range(number_of_windows):

                window_number += 1

                start = index * MONITORING_WINDOW
                end = start + MONITORING_WINDOW

                window = block.iloc[
                    start:end
                ].copy()

                current = window.drop(
                    columns=["_target_time", "_position"]
                ).reset_index(drop=True)

                timestamps = pd.to_datetime(
                    window["_target_time"]
                )

                start_position = int(
                    window["_position"].iloc[0]
                )

                current_start_time = timestamps.iloc[0]

                recent_reference = get_recent_reference(
                    prepared,
                    start_position,
                    current_start_time,
                )

                # FIXED REFERENCE COMPARISON
                fixed_results = evaluate_sensor_groups(
                    reference,
                    current,
                    groups,
                )

                fixed_alert_count = int(
                    fixed_results["group_alert"].sum()
                )

                fixed_alert = (
                    fixed_alert_count >= MIN_DRIFTED_GROUPS
                )

                # RECENT REFERENCE COMPARISON
                recent_alert = None
                recent_alert_count = None

                if recent_reference is not None:

                    recent_results = evaluate_sensor_groups(
                        recent_reference,
                        current,
                        groups,
                    )

                    recent_alert_count = int(
                        recent_results["group_alert"].sum()
                    )

                    recent_alert = (
                        recent_alert_count >= MIN_DRIFTED_GROUPS
                    )

                    recent_results["target"] = target
                    recent_results["window_number"] = window_number
                    recent_results["reference_type"] = "recent"

                    all_group_results.append(
                        recent_results
                    )

                fixed_results["target"] = target
                fixed_results["window_number"] = window_number
                fixed_results["reference_type"] = "fixed"

                all_group_results.append(
                    fixed_results
                )

                all_window_results.append({

                    "target": target,

                    "window_number": window_number,

                    "window_start": timestamps.iloc[0],

                    "window_end": timestamps.iloc[-1],

                    "observations": len(window),

                    "fixed_drifted_groups": fixed_alert_count,

                    "fixed_alert": fixed_alert,

                    "recent_drifted_groups": recent_alert_count,

                    "recent_alert": recent_alert,

                    "recent_reference_available": (
                        recent_reference is not None
                    ),
                })

        print(
            "Monitoring windows evaluated:",
            window_number
        )

    if not all_window_results:
        raise RuntimeError(
            "No monitoring windows were generated."
        )

    windows_df = pd.DataFrame(
        all_window_results
    )

    groups_df = pd.concat(
        all_group_results,
        ignore_index=True
    )

    summary_rows = []

    for target, group in windows_df.groupby("target"):

        recent_available = group[
            group["recent_reference_available"]
        ]

        summary_rows.append({

            "target": target,

            "windows": len(group),

            "fixed_alerts": int(
                group["fixed_alert"].sum()
            ),

            "fixed_alert_rate_pct": (
                100 * group["fixed_alert"].mean()
            ),

            "recent_reference_windows": len(
                recent_available
            ),

            "recent_alerts": int(
                recent_available["recent_alert"].sum()
            ),

            "recent_alert_rate_pct": (
                100 * recent_available["recent_alert"].mean()
                if len(recent_available)
                else np.nan
            ),

            "mean_fixed_drifted_groups": (
                group["fixed_drifted_groups"].mean()
            ),

            "mean_recent_drifted_groups": (
                recent_available[
                    "recent_drifted_groups"
                ].mean()
                if len(recent_available)
                else np.nan
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    group_summary = (
        groups_df.groupby(
            ["target", "reference_type", "sensor_group"]
        )
        .agg(

            windows=("group_alert", "count"),

            alert_rate_pct=(
                "group_alert",
                lambda x: 100 * x.mean()
            ),

            mean_flagged_features=(
                "flagged_features",
                "mean"
            ),

            median_psi=(
                "maximum_psi",
                "median"
            ),

            maximum_psi=(
                "maximum_psi",
                "max"
            ),

            median_minimum_q=(
                "minimum_q_value",
                "median"
            ),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    paths = {

        "windows": (
            OUTPUT_DIR /
            "metropt_temporal_drift_windows.csv"
        ),

        "groups": (
            OUTPUT_DIR /
            "metropt_temporal_drift_groups.csv"
        ),

        "summary": (
            OUTPUT_DIR /
            "metropt_temporal_drift_summary.csv"
        ),
    }

    windows_df.to_csv(
        paths["windows"],
        index=False
    )

    groups_df.to_csv(
        paths["groups"],
        index=False
    )

    summary.to_csv(
        paths["summary"],
        index=False
    )

    print("\n========== FIXED VS RECENT REFERENCE ==========")

    print(
        summary.round(4).to_string(index=False)
    )

    print("\n========== SENSOR-GROUP ALERT RATES ==========")

    print(
        group_summary.sort_values(
            ["target", "reference_type", "alert_rate_pct"],
            ascending=[True, True, False],
        ).round(4).to_string(index=False)
    )

    print("\n========== OUTPUT FILES ==========")

    for path in paths.values():
        print(path)

    print(
        "\nINTERPRETATION CAUTION: These are descriptive alert "
        "frequencies, not measured false-positive rates. Recent "
        "reference comparisons require 500 immediately preceding "
        "consecutive observations. KS p-values remain sensitive "
        "to temporal dependence."
    )


if __name__ == "__main__":
    main()
