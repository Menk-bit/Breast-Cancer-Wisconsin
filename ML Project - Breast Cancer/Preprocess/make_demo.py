import csv
from pathlib import Path

INPUT_PATH = Path("/Users/minhdt/Desktop/ML Breast/Preprocess/export.csv")
OUTPUT_PATH = Path("/Users/minhdt/Desktop/ML Breast/Preprocess/export_alive_>120.csv")

VITAL_STATUS_COL = "Vital status recode (study cutoff used)"
SURVIVAL_MONTHS_COL = "Survival months"

TARGET_STATUS = "Alive"
MIN_SURVIVAL_MONTHS = 120

def find_column_index(header, column_name):
    if column_name in header:
        return header.index(column_name)

    normalized_header = [col.strip().lower() for col in header]
    normalized_col = column_name.strip().lower()

    if normalized_col in normalized_header:
        return normalized_header.index(normalized_col)

    raise ValueError(
        f"Không tìm thấy cột '{column_name}'.\n"
        f"Các cột hiện có là:\n{header}"
    )

def is_dead(value):
    return str(value).strip().lower() == TARGET_STATUS.lower()

def parse_survival_months(value):
    try:
        return int(str(value).strip())
    except ValueError:
        return None

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_PATH.resolve()}")

    total_input_rows = 0
    rows_written = 0
    invalid_survival_months = 0

    with open(INPUT_PATH, "r", encoding="utf-8", newline="") as infile, \
         open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)

        vital_status_idx = find_column_index(header, VITAL_STATUS_COL)
        survival_months_idx = find_column_index(header, SURVIVAL_MONTHS_COL)

        writer.writerow(header)

        for row in reader:
            total_input_rows += 1

            if len(row) <= max(vital_status_idx, survival_months_idx):
                continue

            survival_months = parse_survival_months(row[survival_months_idx])

            if survival_months is None:
                invalid_survival_months += 1
                continue

            if is_dead(row[vital_status_idx]) and survival_months > MIN_SURVIVAL_MONTHS:
                writer.writerow(row)
                rows_written += 1

    print("=" * 80)
    print(f"Created: {OUTPUT_PATH.resolve()}")
    print(f"Total input data rows scanned: {total_input_rows}")
    print(f"Rows written to file: {rows_written}")
    print(f"Total rows including header: {rows_written + 1}")
    print(f"Invalid Survival months rows skipped: {invalid_survival_months}")
    print("=" * 80)

if __name__ == "__main__":
    main()