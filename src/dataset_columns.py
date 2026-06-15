"""Print the columns in one of the repository's model-ready datasets."""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "model_ready_scaled.csv"


def main() -> None:
    for column in pd.read_csv(DATA_PATH, nrows=0).columns:
        print(column)


if __name__ == "__main__":
    main()
