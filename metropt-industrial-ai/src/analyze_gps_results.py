"""Analyze high-speed MetroPT GPS observations and generate a report."""

import numpy as np
import pandas as pd
from pathlib import Path

from quality_checks import (
    calculate_gps_agreement,
    group_high_speed_episodes,
)


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "outputs" / "metropt_gps_speed_consistency.csv"
OUTPUT = BASE / "outputs"
DOCS = BASE / "docs"

OUTPUT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"GPS consistency results not found: {SOURCE}. "
        "Run investigate_gps_consistency.py first."
    )

df = pd.read_csv(SOURCE, parse_dates=["timestamp"])

if df.empty:
    raise ValueError("GPS consistency results contain no observations.")

# Reusable comparison logic.
analysis = calculate_gps_agreement(
    df,
    tolerances=(10, 25, 50),
)

df = analysis["comparison_data"]

valid = df.dropna(
    subset=["reported_speed_kmh", "available_coordinate_mean"]
)

# Percentage difference is descriptive; zero reported speeds are excluded.
df["percentage_difference"] = (
    df["absolute_difference"]
    / df["reported_speed_kmh"].replace(0, np.nan)
) * 100

for tolerance in (10, 25, 50):
    df[f"within_{tolerance}_kmh"] = (
        df["absolute_difference"] <= tolerance
    )

# Reusable episode grouping.
df = group_high_speed_episodes(
    df,
    timestamp_col="timestamp",
    gap_seconds=1,
)

episode_summary = (
    df.groupby("episode_id")
    .agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        observations=("timestamp", "size"),
        maximum_reported_speed=("reported_speed_kmh", "max"),
        mean_reported_speed=("reported_speed_kmh", "mean"),
        maximum_coordinate_speed=("available_coordinate_mean", "max"),
        mean_coordinate_speed=("available_coordinate_mean", "mean"),
        observations_with_comparison=(
            "available_coordinate_mean",
            "count",
        ),
    )
    .reset_index()
)

episode_summary["duration_seconds"] = (
    episode_summary["end"] - episode_summary["start"]
).dt.total_seconds()

episode_path = OUTPUT / "metropt_gps_high_speed_episodes.csv"
episode_summary.to_csv(episode_path, index=False)

# Overall statistics.
correlation = analysis["correlation"]
agreement = {
    int(tolerance): count
    for tolerance, count in analysis["agreement_counts"].items()
}

comparison_distribution = (
    df["valid_comparisons"].value_counts().sort_index().to_dict()
    if "valid_comparisons" in df.columns
    else {
        0: int(df["available_coordinate_mean"].isna().sum()),
        1: int(
            (
                df["previous_coordinate_speed_kmh"].notna()
                ^ df["next_coordinate_speed_kmh"].notna()
            ).sum()
        ),
        2: int(
            (
                df["previous_coordinate_speed_kmh"].notna()
                & df["next_coordinate_speed_kmh"].notna()
            ).sum()
        ),
    }
)

longest_episode = (
    int(episode_summary["observations"].max())
    if not episode_summary.empty else 0
)

maximum_coordinate_speed = (
    valid["available_coordinate_mean"].max()
    if not valid.empty else np.nan
)

report = f"""# MetroPT GPS Speed Quality Assessment

## Dataset scope

- High-speed threshold: >100 km/h
- High-speed observations: {len(df)}
- Observations with at least one valid coordinate comparison: {len(valid)}
- Observations without a valid comparison: {len(df) - len(valid)}
- Maximum allowed adjacent time gap during coordinate analysis: 5 seconds

## Reported GPS speed

{df['reported_speed_kmh'].describe().round(3).to_string()}

## Coordinate-derived speed

{valid['available_coordinate_mean'].describe().round(3).to_string()}

## Comparison

- Pearson correlation: {correlation:.4f}
- Within 10 km/h: {agreement[10]}
- Within 25 km/h: {agreement[25]}
- Within 50 km/h: {agreement[50]}

Agreement counts use only observations with at least one valid
coordinate comparison (n={len(valid)}).

Number of valid comparisons per observation:
{comparison_distribution}

Agreement counts are descriptive sensitivity checks, not validated
GPS accuracy thresholds. Coordinate-derived speed is not independent
ground truth.

## High-speed episodes

- Number of episodes: {len(episode_summary)}
- Longest episode: {longest_episode} observations
- Maximum reported speed: {df['reported_speed_kmh'].max():.2f} km/h
- Maximum coordinate-derived speed: {maximum_coordinate_speed:.2f} km/h

## Preliminary interpretation

The high reported GPS speeds are not independently validated by
coordinate-derived estimates. Some coordinate-derived speeds are
also unusually high.

Possible explanations include GPS positional jumps, measurement
noise, sampling effects, or genuine movement. The current analysis
does not establish which explanation is correct.

GPS observations should not be automatically deleted based only
on a speed threshold.

## Recommended handling

1. Preserve the original GPS measurements.
2. Retain GPS quality indicators.
3. Flag suspicious speed and coordinate transitions.
4. Avoid treating coordinate-derived speed as ground truth.
5. Conduct additional route and positional consistency checks
   before using GPS features in predictive modelling.
"""

report_path = DOCS / "METROPT_GPS_QUALITY_ASSESSMENT.md"
report_path.write_text(report, encoding="utf-8")

print("\n========== GPS QUALITY SUMMARY ==========")
print(f"High-speed observations: {len(df)}")
print(f"Valid coordinate comparisons: {len(valid)}")
print(f"No valid comparison: {len(df) - len(valid)}")
print(f"High-speed episodes: {len(episode_summary)}")

print("\n========== SPEED AGREEMENT ==========")

for tolerance in (10, 25, 50):
    print(
        f"Within {tolerance} km/h: "
        f"{agreement[tolerance]}/{len(valid)}"
    )

print("\nValid comparison count distribution:")
print(comparison_distribution)

print(f"\nPearson correlation: {correlation:.4f}")

print("\n========== TOP 15 HIGH-SPEED EPISODES ==========")
print(
    episode_summary.sort_values(
        "maximum_reported_speed", ascending=False
    ).head(15).to_string(index=False)
)

print("\n========== GENERATED FILES ==========")
print(episode_path)
print(report_path)

print("\nGPS assessment completed.")
