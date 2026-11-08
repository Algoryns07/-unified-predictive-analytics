import pandas as pd
from pathlib import Path

path = Path("metropt-industrial-ai/data/raw/dataset_train.csv")

columns = [
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "LPS",
    "Pressure_switch",
    "gpsSpeed",
    "gpsQuality",
    "gpsLat",
    "gpsLong",
]

exception_rows = []
high_speed_rows = []
total_rows = 0

for chunk in pd.read_csv(
    path,
    usecols=columns,
    chunksize=100_000
):
    total_rows += len(chunk)

    # Investigate the 68 records where both signals are active.
    exceptions = chunk[
        (chunk["COMP"] == 1) &
        (chunk["DV_eletric"] == 1)
    ]

    if not exceptions.empty:
        exception_rows.append(exceptions)

    # Investigate unusually high speeds only when GPS quality is valid.
    high_speed = chunk[
        (chunk["gpsQuality"] == 1) &
        (chunk["gpsSpeed"] > 100)
    ]

    if not high_speed.empty:
        high_speed_rows.append(high_speed)

print("\n========== DATASET ==========")
print(f"Rows scanned: {total_rows:,}")

if exception_rows:
    exceptions = pd.concat(exception_rows, ignore_index=True)

    print("\n========== COMP/DV EXCEPTIONS ==========")
    print(f"Number of records: {len(exceptions)}")
    print(exceptions.to_string(index=False))
else:
    print("\nNo COMP/DV exceptions found.")

if high_speed_rows:
    speeds = pd.concat(high_speed_rows, ignore_index=True)

    print("\n========== HIGH GPS SPEEDS ==========")
    print(f"Records above 100 km/h with valid GPS: {len(speeds)}")
    print(speeds.to_string(index=False))
else:
    print("\nNo high-speed records with valid GPS found.")
