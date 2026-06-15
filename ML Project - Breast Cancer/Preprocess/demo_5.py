import csv
import random
from pathlib import Path

INPUT_PATH = Path("/Users/minhdt/Desktop/ML Breast/Preprocess/export.csv")
OUTPUT_PATH = Path("/Users/minhdt/Desktop/ML Breast/Preprocess/final_demo.csv")

VITAL_STATUS_COL = "Vital status recode (study cutoff used)"
SURVIVAL_MONTHS_COL = "Survival months"

N_ROWS = 200_000

def find_column_index(header, column_name):
    if column_name in header:
        return header.index(column_name)

    normalized_header = [col.strip().lower() for col in header]
    normalized_col = column_name.strip().lower()

    if normalized_col in normalized_header:
        return normalized_header.index(normalized_col)

    raise ValueError(f"Không tìm thấy cột '{column_name}'")

def parse_survival_months(value):
    try:
        return int(str(value).strip())
    except ValueError:
        return None

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_PATH.resolve()}")

    total_input_rows = 0
    valid_rows = []

    with open(INPUT_PATH, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        
        vital_status_idx = find_column_index(header, VITAL_STATUS_COL)
        survival_months_idx = find_column_index(header, SURVIVAL_MONTHS_COL)
        
        header.append("survive_after_5")

        for row in reader:
            total_input_rows += 1

            if len(row) <= max(vital_status_idx, survival_months_idx):
                continue

            survival_months = parse_survival_months(row[survival_months_idx])
            vital_status = str(row[vital_status_idx]).strip().lower()

            if survival_months is None:
                continue

            if survival_months >= 60:
                survive_after_5 = 1
            else:
                if vital_status == "alive":
                    continue
                elif vital_status == "dead":
                    survive_after_5 = 0
                else:
                    continue

            row.append(survive_after_5)
            valid_rows.append(row)

    sample_size = min(N_ROWS, len(valid_rows))
    sampled_rows = random.sample(valid_rows, sample_size)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(sampled_rows)

    print("=" * 60)
    print(f"Created: {OUTPUT_PATH.resolve()}")
    print(f"Total input rows scanned: {total_input_rows}")
    print(f"Total valid rows before sampling: {len(valid_rows)}")
    print(f"Rows written to file: {sample_size}")
    print("=" * 60)

if __name__ == "__main__":
    main()