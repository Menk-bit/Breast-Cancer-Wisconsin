"""Shared dataset loading, preprocessing, and persisted 70/30 split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_HANDLE = "uciml/breast-cancer-wisconsin-data"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "breast-cancer-wisconsin-data.csv"
PROCESSED_DIR = PROJECT_DIR / "artifacts" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
TARGET_COLUMN = "diagnosis"
ID_COLUMN = "sample_id"
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_SPLITS = 5


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series


def find_dataset() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH

    local_candidates = (
        PROJECT_DIR / "data.csv",
        PROJECT_DIR / "breast-cancer-wisconsin-data_data.csv",
    )
    for candidate in local_candidates:
        if candidate.exists():
            return candidate

    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    matches = list(dataset_dir.rglob("data.csv"))
    if not matches:
        raise FileNotFoundError(f"data.csv was not found in {dataset_dir}")
    return matches[0]


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    if "id" in cleaned:
        cleaned = cleaned.rename(columns={"id": ID_COLUMN})
    elif ID_COLUMN not in cleaned:
        cleaned.insert(0, ID_COLUMN, cleaned.index)

    cleaned = cleaned.drop(columns=["Unnamed: 32"], errors="ignore")
    cleaned = cleaned.dropna(axis=1, how="all")
    if TARGET_COLUMN not in cleaned:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")

    label_map = {"B": 0, "M": 1, "Benign": 0, "Malignant": 1}
    mapped = cleaned[TARGET_COLUMN].map(label_map)
    numeric = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned[TARGET_COLUMN] = mapped.where(mapped.notna(), numeric)
    if (
        cleaned[TARGET_COLUMN].isna().any()
        or not cleaned[TARGET_COLUMN].isin([0, 1]).all()
    ):
        raise ValueError("Diagnosis labels must be B/M, Benign/Malignant, or 0/1.")
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    feature_columns = cleaned.columns.drop([ID_COLUMN, TARGET_COLUMN])
    cleaned[feature_columns] = cleaned[feature_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    return cleaned


def prepare_data(force_split: bool = False) -> DataSplit:
    """Return the one shared stratified holdout used by every algorithm."""
    if force_split or not (TRAIN_PATH.exists() and TEST_PATH.exists()):
        source_path = find_dataset()
        cleaned = clean_data(pd.read_csv(source_path))
        train, test = train_test_split(
            cleaned,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=cleaned[TARGET_COLUMN],
        )
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train.to_csv(TRAIN_PATH, index=False)
        test.to_csv(TEST_PATH, index=False)
        metadata = {
            "dataset": DATASET_HANDLE,
            "source_file": str(source_path),
            "target": TARGET_COLUMN,
            "id_column": ID_COLUMN,
            "label_mapping": {"B": 0, "M": 1},
            "train_rows": len(train),
            "test_rows": len(test),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_splits": CV_SPLITS,
            "feature_columns": [
                column
                for column in cleaned.columns
                if column not in (ID_COLUMN, TARGET_COLUMN)
            ],
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        if ID_COLUMN not in train or ID_COLUMN not in test:
            return prepare_data(force_split=True)

    feature_columns = [
        column
        for column in train.columns
        if column not in (ID_COLUMN, TARGET_COLUMN)
    ]
    return DataSplit(
        X_train=train[feature_columns],
        X_test=test[feature_columns],
        y_train=train[TARGET_COLUMN],
        y_test=test[TARGET_COLUMN],
        train_ids=train[ID_COLUMN],
        test_ids=test[ID_COLUMN],
    )


def main() -> None:
    split = prepare_data(force_split=True)
    print(f"Training set: {split.X_train.shape}")
    print(f"Test set:     {split.X_test.shape}")
    print(f"Saved shared split to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
