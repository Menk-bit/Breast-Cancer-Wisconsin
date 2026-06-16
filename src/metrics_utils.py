"""Shared binary-classification metrics and threshold selection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_COLUMNS = [
    "accuracy",
    "precision_class_0",
    "precision_class_1",
    "recall_class_0",
    "recall_class_1",
    "f1_class_0",
    "f1_class_1",
    "roc_auc",
]


def roc_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Return binary ROC AUC as a plain float."""
    return float(roc_auc_score(y_true, y_score))


def binary_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float],
) -> dict[str, float]:
    """Calculate the project's standard binary-classification metrics."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision_class_0": float(
            precision_score(
                y_true_array,
                y_pred_array,
                labels=[0, 1],
                pos_label=0,
                zero_division=0,
            )
        ),
        "precision_class_1": float(
            precision_score(
                y_true_array,
                y_pred_array,
                labels=[0, 1],
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_class_0": float(
            recall_score(
                y_true_array,
                y_pred_array,
                labels=[0, 1],
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_class_1": float(
            recall_score(
                y_true_array,
                y_pred_array,
                labels=[0, 1],
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_class_0": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=[0, 1],
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_class_1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=[0, 1],
                pos_label=1,
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc(y_true_array, y_score_array),
    }


def evaluate_scores(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Threshold scores and return standard metrics plus predictions."""
    y_score_array = np.asarray(y_score, dtype=float)
    predictions = (y_score_array >= threshold).astype(np.int8)
    metrics = binary_classification_metrics(y_true, predictions, y_score_array)
    return {"threshold": float(threshold), **metrics}, predictions


def select_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select a validation threshold by class-1 F1, then class-0 F1 and ROC-AUC."""
    candidates = (
        np.linspace(0.10, 0.90, 161)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=float)
    )
    rows = []
    for threshold in candidates:
        metrics, _ = evaluate_scores(y_true, y_score, float(threshold))
        rows.append(metrics)

    frame = pd.DataFrame(rows).sort_values(
        ["f1_class_1", "f1_class_0", "roc_auc"],
        ascending=False,
    )
    return float(frame.iloc[0]["threshold"]), frame.reset_index(drop=True)


def select_threshold_for_class_0(
    y_true: Iterable[int],
    y_score: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select a threshold prioritizing class-0 recall, then class-0 F1."""
    candidates = (
        np.linspace(0.05, 0.95, 181)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=float)
    )
    rows = []
    for threshold in candidates:
        metrics, _ = evaluate_scores(y_true, y_score, float(threshold))
        rows.append(metrics)

    frame = pd.DataFrame(rows).sort_values(
        ["recall_class_0", "f1_class_0", "roc_auc"],
        ascending=False,
    )
    return float(frame.iloc[0]["threshold"]), frame.reset_index(drop=True)


def maximum_f1_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    positive_class: int = 1,
) -> dict[str, object]:
    """Select the score threshold with the highest class-1 F1."""
    y_binary = (np.asarray(y_true) == positive_class).astype(int)
    scores = np.asarray(y_score, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_binary, scores)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "roc_auc": roc_auc(y_binary, scores),
        "curve": pd.DataFrame(
            {
                "threshold": thresholds,
                "precision_class_1": precision,
                "recall_class_1": recall,
                "f1_class_1": f1,
            }
        ),
    }
