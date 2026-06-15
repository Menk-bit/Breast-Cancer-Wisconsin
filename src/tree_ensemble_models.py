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
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data_splitting import (
    stratified_sample,
    stratified_train_validation_test_split,
)
from metrics_utils import METRIC_COLUMNS, evaluate_scores, select_threshold

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
        X, y = stratified_sample(X, y, sample_size, RANDOM_STATE)

    return X.reset_index(drop=True), y.reset_index(drop=True)


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

    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data.resolve(), args.sample_size)
    split = stratified_train_validation_test_split(X, y, RANDOM_STATE)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split
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
        threshold, threshold_metrics = select_threshold(
            y_valid, valid_probabilities
        )
        test_probabilities = model.predict_proba(X_test)[:, 1]
        metrics, predictions = evaluate_scores(
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
        threshold_metrics.to_csv(
            output_dir / name / "validation_threshold_metrics.csv",
            index=False,
        )
        summary_rows.append({"model": name, **metrics})
        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1_class_1={metrics['f1_class_1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"threshold={threshold:.3f}, time={training_seconds:.1f}s"
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["f1_class_1", "roc_auc"], ascending=False
    )
    summary.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\nSaved results to {output_dir}")
    print(
        summary[
            [
                "model",
                *METRIC_COLUMNS,
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
