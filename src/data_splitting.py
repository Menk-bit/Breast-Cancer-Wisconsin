"""Shared stratified sampling and train/validation/test splitting."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from sklearn.model_selection import train_test_split


TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_STATE = 42


class DataSplit(NamedTuple):
    X_train: Any
    X_validation: Any
    X_test: Any
    y_train: Any
    y_validation: Any
    y_test: Any


def stratified_train_validation_test_split(
    X: Any,
    y: Any,
    random_state: int = RANDOM_STATE,
) -> DataSplit:
    """Split X/y into stratified train, validation, and test sets at 60/20/20."""
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=random_state,
    )
    validation_fraction = VALIDATION_RATIO / (TRAIN_RATIO + VALIDATION_RATIO)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_fraction,
        stratify=y_train_validation,
        random_state=random_state + 1,
    )
    return DataSplit(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def stratified_sample(
    X: Any,
    y: Any,
    max_rows: int | None,
    random_state: int = RANDOM_STATE,
) -> tuple[Any, Any]:
    """Return a reproducible stratified sample without changing input types."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2.")

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        stratify=y,
        random_state=random_state,
    )
    return X_sample, y_sample


def stratified_two_way_split(
    X: Any,
    y: Any,
    *,
    test_size: float,
    random_state: int = RANDOM_STATE,
) -> tuple[Any, Any, Any, Any]:
    """Return a reproducible stratified two-way split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def split_class_distribution(y: Any) -> dict[int, int]:
    """Return class counts for concise split diagnostics."""
    labels, counts = np.unique(np.asarray(y), return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}
