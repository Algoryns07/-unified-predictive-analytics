import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from quality_checks import (
    calculate_gps_agreement,
    find_signal_overlap_transitions,
    group_high_speed_episodes,
    identify_timestamp_gaps,
)


def test_detect_signal_overlap_transition():
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2022-01-01", periods=3, freq="s"
        ),
        "COMP": [0, 1, 1],
        "DV_eletric": [1, 1, 0],
    })

    result = find_signal_overlap_transitions(df)

    assert len(result) == 1
    assert result.iloc[0]["transition"] == "0/1 -> 1/1 -> 1/0"


def test_ignore_overlap_without_both_neighbours():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-01-01 00:00:00"]),
        "COMP": [1],
        "DV_eletric": [1],
    })

    assert find_signal_overlap_transitions(df).empty


def test_reject_missing_signal_columns():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2022-01-01"])})

    with pytest.raises(ValueError, match="Missing required columns"):
        find_signal_overlap_transitions(df)


def test_reject_non_binary_signal_values():
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2022-01-01", periods=3, freq="s"
        ),
        "COMP": [0, 2, 1],
        "DV_eletric": [1, 1, 0],
    })

    with pytest.raises(ValueError, match="binary"):
        find_signal_overlap_transitions(df)


def test_reject_duplicate_signal_timestamps():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-01-01",
            "2022-01-01",
        ]),
        "COMP": [1, 1],
        "DV_eletric": [0, 1],
    })

    with pytest.raises(ValueError, match="Duplicate"):
        find_signal_overlap_transitions(df)


def gps_frame():
    return pd.DataFrame({
        "reported_speed_kmh": [100, 100, 100, 100],
        "previous_coordinate_speed_kmh": [95, np.nan, 80, np.inf],
        "next_coordinate_speed_kmh": [105, 110, np.nan, 90],
    })


def test_gps_agreement_uses_valid_comparisons():
    result = calculate_gps_agreement(gps_frame())

    assert result["total_observations"] == 4
    assert result["valid_observations"] == 4
    assert result["missing_observations"] == 0
    assert result["agreement_counts"][10.0] == 3


def test_gps_missing_comparisons_do_not_count_as_agreement():
    df = pd.DataFrame({
        "reported_speed_kmh": [100],
        "previous_coordinate_speed_kmh": [np.nan],
        "next_coordinate_speed_kmh": [np.nan],
    })

    result = calculate_gps_agreement(df)

    assert result["valid_observations"] == 0
    assert result["agreement_counts"][10.0] == 0


def test_gps_numeric_strings_are_converted():
    df = pd.DataFrame({
        "reported_speed_kmh": ["100", "bad"],
        "previous_coordinate_speed_kmh": ["95", "90"],
        "next_coordinate_speed_kmh": [None, None],
    })

    result = calculate_gps_agreement(df)

    assert result["valid_observations"] == 1
    assert result["comparison_data"]["reported_speed_kmh"].iloc[0] == 100


def test_gps_infinite_values_are_excluded():
    df = pd.DataFrame({
        "reported_speed_kmh": [100, np.inf],
        "previous_coordinate_speed_kmh": [95, 90],
        "next_coordinate_speed_kmh": [np.nan, np.nan],
    })

    result = calculate_gps_agreement(df)

    assert result["valid_observations"] == 1


def test_reject_missing_gps_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_gps_agreement(pd.DataFrame({"reported_speed_kmh": [10]}))


@pytest.mark.parametrize("tolerances", [
    (),
    (-1,),
    (np.inf,),
    (np.nan,),
    (True,),
])
def test_reject_invalid_gps_tolerances(tolerances):
    with pytest.raises(ValueError):
        calculate_gps_agreement(gps_frame(), tolerances=tolerances)


def test_group_consecutive_high_speed_observations():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-01-01 00:00:00",
            "2022-01-01 00:00:01",
            "2022-01-01 00:00:05",
        ])
    })

    result = group_high_speed_episodes(df)

    assert result["episode_id"].tolist() == [1, 1, 2]


def test_episode_grouping_sorts_timestamps():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-01-01 00:00:05",
            "2022-01-01 00:00:00",
            "2022-01-01 00:00:01",
        ])
    })

    result = group_high_speed_episodes(df)

    assert result["timestamp"].is_monotonic_increasing
    assert result["episode_id"].tolist() == [1, 1, 2]


@pytest.mark.parametrize("gap", [0, -1, np.inf, np.nan, True])
def test_reject_invalid_episode_gap(gap):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-01-01"])
    })

    with pytest.raises(ValueError):
        group_high_speed_episodes(df, gap_seconds=gap)


def test_reject_invalid_episode_timestamp():
    df = pd.DataFrame({"timestamp": ["not-a-date"]})

    with pytest.raises(ValueError, match="invalid"):
        group_high_speed_episodes(df)


def test_identify_timestamp_gaps():
    timestamps = pd.to_datetime([
        "2022-01-01 00:00:00",
        "2022-01-01 00:00:01",
        "2022-01-01 00:00:05",
    ])

    result = identify_timestamp_gaps(timestamps)

    assert len(result) == 1
    assert result.iloc[0]["gap_seconds"] == 4


def test_regular_sampling_has_no_gaps():
    timestamps = pd.date_range(
        "2022-01-01", periods=5, freq="s"
    )

    assert identify_timestamp_gaps(timestamps).empty


def test_timestamp_gap_detection_sorts_input():
    timestamps = pd.to_datetime([
        "2022-01-01 00:00:05",
        "2022-01-01 00:00:00",
        "2022-01-01 00:00:01",
    ])

    result = identify_timestamp_gaps(timestamps)

    assert len(result) == 1
    assert result.iloc[0]["gap_seconds"] == 4


@pytest.mark.parametrize("interval", [0, -1, np.inf, np.nan, True])
def test_reject_invalid_expected_timestamp_interval(interval):
    with pytest.raises(ValueError):
        identify_timestamp_gaps(
            pd.date_range("2022-01-01", periods=3, freq="s"),
            expected_interval_seconds=interval,
        )


def test_reject_invalid_timestamp_values():
    with pytest.raises(ValueError, match="invalid"):
        identify_timestamp_gaps([
            "2022-01-01",
            "not-a-date",
        ])


def test_duplicate_timestamps_are_not_reported_as_gaps():
    timestamps = pd.to_datetime([
        "2022-01-01 00:00:00",
        "2022-01-01 00:00:00",
        "2022-01-01 00:00:01",
    ])

    assert identify_timestamp_gaps(timestamps).empty
