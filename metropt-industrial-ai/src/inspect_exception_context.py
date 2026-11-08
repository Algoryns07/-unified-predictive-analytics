import pandas as pd
from pathlib import Path

path = Path("metropt-industrial-ai/data/raw/dataset_train.csv")
output = Path("metropt-industrial-ai/outputs")
output.mkdir(parents=True, exist_ok=True)

columns = [
    "timestamp",
    "TP2",
    "TP3",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "LPS",
]

# First pass: locate all contradictory observations.
exceptions = []

for chunk in pd.read_csv(
    path,
    usecols=columns,
    chunksize=100_000
):
    mask = (
        (chunk["COMP"] == 1) &
        (chunk["DV_eletric"] == 1)
    )

    if mask.any():
        exceptions.append(chunk.loc[mask].copy())

if not exceptions:
    raise RuntimeError("No contradictory records found.")

exceptions = pd.concat(exceptions, ignore_index=True)
exceptions["timestamp"] = pd.to_datetime(exceptions["timestamp"])

exception_times = set(exceptions["timestamp"])

# Second pass: retain observations within 10 seconds
# of each contradictory timestamp.
context_parts = []

for chunk in pd.read_csv(
    path,
    usecols=columns,
    chunksize=100_000
):
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])

    for timestamp in exception_times:
        mask = (
            (chunk["timestamp"] >= timestamp - pd.Timedelta(seconds=10)) &
            (chunk["timestamp"] <= timestamp + pd.Timedelta(seconds=10))
        )

        if mask.any():
            context_parts.append(chunk.loc[mask].copy())

context = pd.concat(context_parts, ignore_index=True)
context = context.drop_duplicates(subset=["timestamp"])
context = context.sort_values("timestamp")

context["is_exception"] = (
    (context["COMP"] == 1) &
    (context["DV_eletric"] == 1)
)

destination = output / "metropt_signal_exception_context.csv"
context.to_csv(destination, index=False)

print("\n========== TEMPORAL CONTEXT ==========")
print(f"Contradictory observations: {len(exceptions)}")
print(f"Context observations: {len(context)}")
print(f"Saved to: {destination}")

print("\n========== FIRST 60 CONTEXT RECORDS ==========")
print(context.head(60).to_string(index=False))

print("\nInvestigation completed.")
