import numpy as np
import pandas as pd
from pathlib import Path

SOURCE = Path("metropt-industrial-ai/data/raw/dataset_train.csv")
OUTPUT = Path("metropt-industrial-ai/outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

USECOLS = [
    "timestamp",
    "gpsLong",
    "gpsLat",
    "gpsSpeed",
    "gpsQuality",
]

CHUNK_SIZE = 100_000
HIGH_SPEED = 100.0
MAX_GAP_SECONDS = 5

def haversine_km(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance between GPS coordinates."""
    earth_radius_km = 6371.0088

    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def calculate_segment_speed(first, second):
    """Return coordinate-derived speed and elapsed seconds."""
    seconds = (
        second["timestamp"] - first["timestamp"]
    ).total_seconds()

    valid = (
        seconds > 0
        and seconds <= MAX_GAP_SECONDS
        and first["gpsQuality"] == 1
        and second["gpsQuality"] == 1
        and first["gpsLong"] != 0
        and first["gpsLat"] != 0
        and second["gpsLong"] != 0
        and second["gpsLat"] != 0
    )

    if not valid:
        return np.nan, seconds

    distance = haversine_km(
        first["gpsLong"],
        first["gpsLat"],
        second["gpsLong"],
        second["gpsLat"],
    )

    return distance / seconds * 3600, seconds


results = []
carry = None
rows_scanned = 0
candidates_found = 0

for chunk in pd.read_csv(
    SOURCE,
    usecols=USECOLS,
    chunksize=CHUNK_SIZE,
):
    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"], errors="coerce"
    )

    rows_scanned += len(chunk)

    if carry is not None:
        frame = pd.concat(
            [carry, chunk],
            ignore_index=True,
        )
    else:
        frame = chunk.reset_index(drop=True)

    # Hold the last row until the next chunk so that
    # every candidate can have a following observation.
    if len(frame) < 2:
        carry = frame.copy()
        continue

    process = frame.iloc[:-1].reset_index(drop=True)

    for position in range(len(process)):
        current = process.iloc[position]

        if not (
            current["gpsQuality"] == 1
            and current["gpsSpeed"] > HIGH_SPEED
        ):
            continue

        candidates_found += 1

        previous = (
            process.iloc[position - 1]
            if position > 0
            else None
        )

        following = frame.iloc[position + 1]

        if previous is not None:
            previous_speed, previous_gap = (
                calculate_segment_speed(previous, current)
            )
        else:
            previous_speed, previous_gap = np.nan, np.nan

        next_speed, next_gap = calculate_segment_speed(
            current, following
        )

        results.append({
            "timestamp": current["timestamp"],
            "reported_speed_kmh": current["gpsSpeed"],
            "gps_quality": current["gpsQuality"],
            "longitude": current["gpsLong"],
            "latitude": current["gpsLat"],
            "previous_gap_seconds": previous_gap,
            "previous_coordinate_speed_kmh": previous_speed,
            "next_gap_seconds": next_gap,
            "next_coordinate_speed_kmh": next_speed,
        })

    carry = frame.iloc[[-1]].copy()


result = pd.DataFrame(results)

if result.empty:
    raise RuntimeError("No high-speed GPS observations found.")

result["previous_speed_difference"] = (
    result["reported_speed_kmh"]
    - result["previous_coordinate_speed_kmh"]
)

result["next_speed_difference"] = (
    result["reported_speed_kmh"]
    - result["next_coordinate_speed_kmh"]
)

result["valid_comparisons"] = (
    result["previous_coordinate_speed_kmh"].notna().astype(int)
    + result["next_coordinate_speed_kmh"].notna().astype(int)
)

destination = OUTPUT / "metropt_gps_speed_consistency.csv"
result.to_csv(destination, index=False)

print("\n========== GPS CONSISTENCY INVESTIGATION ==========")
print(f"Rows scanned: {rows_scanned:,}")
print(f"High-speed observations found: {candidates_found}")
print(f"Results saved: {destination}")

print("\n========== COMPARISON COVERAGE ==========")
print(
    result["valid_comparisons"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "No valid adjacent comparison",
        1: "One valid adjacent comparison",
        2: "Two valid adjacent comparisons",
    })
    .to_string()
)

print("\n========== REPORTED GPS SPEED ==========")
print(result["reported_speed_kmh"].describe().round(3).to_string())

print("\n========== PREVIOUS SEGMENT SPEED ==========")
print(
    result["previous_coordinate_speed_kmh"]
    .describe()
    .round(3)
    .to_string()
)

print("\n========== NEXT SEGMENT SPEED ==========")
print(
    result["next_coordinate_speed_kmh"]
    .describe()
    .round(3)
    .to_string()
)

print("\n========== FIRST 30 OBSERVATIONS ==========")
print(
    result[
        [
            "timestamp",
            "reported_speed_kmh",
            "previous_coordinate_speed_kmh",
            "next_coordinate_speed_kmh",
            "valid_comparisons",
        ]
    ].head(30).round(3).to_string(index=False)
)

print("\nInvestigation completed.")
