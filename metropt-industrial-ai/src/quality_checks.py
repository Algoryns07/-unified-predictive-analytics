"""Reusable, validated MetroPT data-quality analysis functions."""

import numpy as np
import pandas as pd


def _require_columns(df, required):
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _validate_positive_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number.")


def find_signal_overlap_transitions(
    df,
    comp_col="COMP",
    dv_col="DV_eletric",
    timestamp_col="timestamp",
):
    """Identify simultaneous signal states with exact one-second neighbours."""

    _require_columns(df, {comp_col, dv_col, timestamp_col})

    data = df.copy()
    data[timestamp_col] = pd.to_datetime(
        data[timestamp_col], errors="coerce"
    )

    if data[timestamp_col].isna().any():
        raise ValueError("Timestamp column contains invalid values.")

    for column in (comp_col, dv_col):
        values = pd.to_numeric(data[column], errors="coerce")

        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(
                f"{column} must contain only binary values 0 and 1."
            )

        data[column] = values.astype(int)

    data = data.sort_values(timestamp_col).reset_index(drop=True)

    if data[timestamp_col].duplicated().any():
        raise ValueError("Duplicate timestamps are not supported.")

    overlap_positions = np.flatnonzero(
        (data[comp_col] == 1) & (data[dv_col] == 1)
    )

    records = []

    for position in overlap_positions:
        if position == 0 or position == len(data) - 1:
            continue

        before = data.iloc[position - 1]
        current = data.iloc[position]
        after = data.iloc[position + 1]

        before_gap = (
            current[timestamp_col] - before[timestamp_col]
        ).total_seconds()

        after_gap = (
            after[timestamp_col] - current[timestamp_col]
        ).total_seconds()

        if before_gap != 1 or after_gap != 1:
            continue

        before_state = (int(before[comp_col]), int(before[dv_col]))
        overlap_state = (int(current[comp_col]), int(current[dv_col]))
        after_state = (int(after[comp_col]), int(after[dv_col]))

        records.append({
            "timestamp": current[timestamp_col],
            "before_state": before_state,
            "overlap_state": overlap_state,
            "after_state": after_state,
            "transition": (
                f"{before_state[0]}/{before_state[1]} -> "
                f"{overlap_state[0]}/{overlap_state[1]} -> "
                f"{after_state[0]}/{after_state[1]}"
            ),
        })

    return pd.DataFrame(records)


def calculate_gps_agreement(df, tolerances=(10, 25, 50)):
    """Compare reported GPS speed with available coordinate-derived speeds.

    An observation is valid when its reported speed and at least one
    coordinate-derived speed are finite numeric values.
    """

    required = {
        "reported_speed_kmh",
        "previous_coordinate_speed_kmh",
        "next_coordinate_speed_kmh",
    }

    _require_columns(df, required)

    if not tolerances:
        raise ValueError("At least one tolerance must be supplied.")

    validated_tolerances = []

    for tolerance in tolerances:
        if (
            isinstance(tolerance, bool)
            or not isinstance(
                tolerance, (int, float, np.integer, np.floating)
            )
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError(
                "Tolerances must be finite, non-negative numbers."
            )

        validated_tolerances.append(float(tolerance))

    result = df.copy()

    numeric_columns = [
        "reported_speed_kmh",
        "previous_coordinate_speed_kmh",
        "next_coordinate_speed_kmh",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    previous = result["previous_coordinate_speed_kmh"]
    following = result["next_coordinate_speed_kmh"]
    reported = result["reported_speed_kmh"]

    result["available_coordinate_mean"] = pd.concat(
        [previous, following], axis=1
    ).mean(axis=1)

    result["absolute_difference"] = (
        reported - result["available_coordinate_mean"]
    ).abs()

    valid_mask = (
        reported.notna()
        & result["available_coordinate_mean"].notna()
        & np.isfinite(reported)
        & np.isfinite(result["available_coordinate_mean"])
    )

    valid = result.loc[valid_mask].copy()

    agreement = {}

    for tolerance in validated_tolerances:
        agreement[tolerance] = int(
            (valid["absolute_difference"] <= tolerance).sum()
        )

    correlation = (
        valid["reported_speed_kmh"].corr(
            valid["available_coordinate_mean"]
        )
        if len(valid) > 1 else np.nan
    )

    return {
        "total_observations": len(result),
        "valid_observations": len(valid),
        "missing_observations": len(result) - len(valid),
        "agreement_counts": agreement,
        "correlation": correlation,
        "comparison_data": result,
    }


def group_high_speed_episodes(
    df,
    timestamp_col="timestamp",
    gap_seconds=1,
):
    """Group observations into episodes separated by timestamp gaps."""

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    _validate_positive_number(gap_seconds, "gap_seconds")

    result = df.copy()
    result[timestamp_col] = pd.to_datetime(
        result[timestamp_col], errors="coerce"
    )

    if result[timestamp_col].isna().any():
        raise ValueError("Timestamp column contains invalid values.")

    result = result.sort_values(timestamp_col).reset_index(drop=True)

    gaps = result[timestamp_col].diff().dt.total_seconds()

    result["time_gap_seconds"] = gaps
    result["episode_id"] = (
        gaps.isna() | (gaps > gap_seconds)
    ).cumsum()

    return result


def identify_timestamp_gaps(
    timestamps,
    expected_interval_seconds=1,
):
    """Return sorted timestamp intervals exceeding the expected interval.

    The function sorts timestamps before calculating differences.
    Duplicate timestamps are not reported as gaps.
    """

    _validate_positive_number(
        expected_interval_seconds,
        "expected_interval_seconds",
    )

    values = pd.Series(
        pd.to_datetime(timestamps, errors="coerce")
    ).reset_index(drop=True)

    if values.isna().any():
        raise ValueError("Timestamp input contains invalid values.")

    values = values.sort_values().reset_index(drop=True)

    gaps = values.diff().dt.total_seconds()

    result = pd.DataFrame({
        "previous_timestamp": values.shift(1),
        "current_timestamp": values,
        "gap_seconds": gaps,
    })

    return result[
        result["gap_seconds"] > expected_interval_seconds
    ].reset_index(drop=True)
