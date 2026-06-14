import csv
from pathlib import Path


INPUT_PATH = Path("/Users/minhdt/Desktop/ML Breast/export.csv")
OUTPUT_PATH = Path("export_demo.csv")

N_ROWS = 200_000  # số dòng data, không tính header


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_PATH.resolve()}")

    with open(INPUT_PATH, "r", encoding="utf-8", newline="") as infile, \
         open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Ghi header
        header = next(reader)
        writer.writerow(header)

        # Ghi 200,000 dòng đầu tiên
        count = 0
        for row in reader:
            if count >= N_ROWS:
                break

            writer.writerow(row)
            count += 1

    print(f"Created: {OUTPUT_PATH.resolve()}")
    print(f"Number of data rows written: {count}")
    print(f"Total rows including header: {count + 1}")


if __name__ == "__main__":
    main()