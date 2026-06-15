"""Train and compare Random Forest and XGBoost on the local SEER dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "model_ready_tree.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tree_ensemble_outputs"
TARGET_COLUMN = "survive_after_5"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Random Forest and XGBoost on model_ready_tree.csv."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for a faster run.",
    )
    return parser.parse_args()


def load_data(
    path: Path, sample_size: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    df = df.loc[df[TARGET_COLUMN].notna()].copy()
    y = pd.to_numeric(df.pop(TARGET_COLUMN), errors="raise").astype(np.int8)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1, found: {sorted(y.unique())}")

    leakage_columns = [
        column
        for column in (
            "event_dead",
            "survival_months",
            "survival_months_int",
            "survival_months_unknown_flag",
        )
        if column in df.columns
    ]
    X = df.drop(columns=leakage_columns)
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)

    if sample_size is not None and sample_size < len(X):
        if sample_size < 100:
            raise ValueError("--sample-size must be at least 100.")
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )

    return X.reset_index(drop=True), y.reset_index(drop=True)


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        stratify=y_train_val,
        random_state=RANDOM_STATE + 1,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.10, 0.90, 161)
    scores = [
        balanced_accuracy_score(y_true, probabilities >= threshold)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def calculate_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, float | int], np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics, predictions


def build_models() -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def save_model_outputs(
    name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    metrics: dict[str, float | int],
    output_dir: Path,
) -> None:
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "probability_survive_after_5": probabilities,
            "predicted": predictions,
        }
    ).to_csv(model_dir / "test_predictions.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)

    report = classification_report(
        y_test,
        predictions,
        target_names=["not_survive_5_years", "survive_5_years"],
        digits=4,
        zero_division=0,
    )
    (model_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data.resolve(), args.sample_size)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    print(
        f"Loaded {len(X):,} rows and {X.shape[1]} features "
        f"(positive rate: {y.mean():.3f})."
    )
    print(
        f"Split sizes: train={len(X_train):,}, validation={len(X_valid):,}, "
        f"test={len(X_test):,}."
    )

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    summary_rows = []

    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()
        if name == "xgboost":
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start

        valid_probabilities = model.predict_proba(X_valid)[:, 1]
        threshold = select_threshold(y_valid, valid_probabilities)
        test_probabilities = model.predict_proba(X_test)[:, 1]
        metrics, predictions = calculate_metrics(
            y_test, test_probabilities, threshold
        )
        metrics["training_seconds"] = training_seconds
        metrics["rows"] = len(X)
        metrics["features"] = X.shape[1]

        save_model_outputs(
            name,
            model,
            X_test,
            y_test,
            test_probabilities,
            predictions,
            metrics,
            output_dir,
        )
        summary_rows.append({"model": name, **metrics})
        print(
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
            f"threshold={threshold:.3f}, time={training_seconds:.1f}s"
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        "balanced_accuracy", ascending=False
    )
    summary.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\nSaved results to {output_dir}")
    print(
        summary[
            [
                "model",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
