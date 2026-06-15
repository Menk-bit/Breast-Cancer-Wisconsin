"""Load the model-ready five-year survival data and create a shared split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_PATHS = {
    "scaled": DATA_DIR / "model_ready_scaled.csv",
    "tree": DATA_DIR / "model_ready_tree.csv",
}
PROCESSED_DIR = PROJECT_DIR / "artifacts" / "processed"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
TARGET_COLUMN = "survive_after_5"
OUTCOME_COLUMNS: tuple[str, ...] = ()
ID_COLUMN = "row_id"
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_SPLITS = 5

DatasetVariant = Literal["scaled", "tree"]


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    dataset_signature: str


def _load_dataset(dataset_variant: DatasetVariant) -> tuple[pd.DataFrame, Path]:
    try:
        source_path = DATA_PATHS[dataset_variant]
    except KeyError as exc:
        choices = ", ".join(DATA_PATHS)
        raise ValueError(
            f"Unknown dataset variant {dataset_variant!r}; choose {choices}."
        ) from exc

    if not source_path.exists():
        raise FileNotFoundError(f"Required dataset was not found: {source_path}")

    data = pd.read_csv(source_path)
    required = {TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{source_path.name} is missing columns: {sorted(missing)}")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values.")
    if not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")

    feature_columns = [
        column
        for column in data.columns
        if column not in (TARGET_COLUMN, *OUTCOME_COLUMNS)
    ]
    if data[feature_columns].isna().any().any():
        missing_features = data[feature_columns].columns[
            data[feature_columns].isna().any()
        ].tolist()
        raise ValueError(f"Feature columns contain missing values: {missing_features}")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int8")
    return data, source_path


def _dataset_signature(
    source_path: Path,
    feature_columns: list[str],
    row_count: int,
) -> str:
    stat = source_path.stat()
    contract = {
        "source": source_path.name,
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "rows": row_count,
        "features": feature_columns,
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_data(dataset_variant: DatasetVariant = "tree") -> DataSplit:
    """Return a deterministic stratified split for one model-ready variant."""
    data, source_path = _load_dataset(dataset_variant)
    feature_columns = [
        column
        for column in data.columns
        if column not in (TARGET_COLUMN, *OUTCOME_COLUMNS)
    ]
    row_ids = pd.Series(data.index, index=data.index, name=ID_COLUMN)
    train_ids, test_ids = train_test_split(
        row_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )
    train_index = train_ids.index
    test_index = test_ids.index
    signature = _dataset_signature(source_path, feature_columns, len(data))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_file": str(source_path),
        "dataset_variant": dataset_variant,
        "dataset_signature": signature,
        "target": TARGET_COLUMN,
        "excluded_outcome_columns": list(OUTCOME_COLUMNS),
        "id_column": ID_COLUMN,
        "rows": len(data),
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "feature_columns": feature_columns,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return DataSplit(
        X_train=data.loc[train_index, feature_columns],
        X_test=data.loc[test_index, feature_columns],
        y_train=data.loc[train_index, TARGET_COLUMN],
        y_test=data.loc[test_index, TARGET_COLUMN],
        train_ids=train_ids,
        test_ids=test_ids,
        dataset_signature=signature,
    )


def main() -> None:
    for variant in DATA_PATHS:
        split = prepare_data(variant)
        print(
            f"{variant:>6}: train={split.X_train.shape}, "
            f"test={split.X_test.shape}"
        )


if __name__ == "__main__":
    main()
