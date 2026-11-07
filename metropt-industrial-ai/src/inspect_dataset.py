import csv
from pathlib import Path

file_path = Path("metropt-industrial-ai/data/raw/dataset_train.csv")

print("=" * 65)
print("METROPT DATASET INSPECTION")
print("=" * 65)

print(f"\nFile: {file_path}")
print(f"Size: {file_path.stat().st_size / (1024**3):.3f} GB")

with file_path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)

    header = next(reader)

    print(f"\nNumber of columns: {len(header)}")
    print("\nCOLUMN NAMES:")
    for i, column in enumerate(header, 1):
        print(f"{i:02d}. {column}")

    print("\nFIRST FIVE RECORDS:")
    for i, row in enumerate(reader):
        if i >= 5:
            break
        print(f"\nRecord {i+1}:")
        for column, value in zip(header, row):
            print(f"  {column}: {value}")

print("\nInspection completed.")
