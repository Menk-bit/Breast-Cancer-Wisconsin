"""From-scratch logistic regression and Gaussian Naive Bayes comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_splitting import (
    stratified_sample,
    stratified_train_validation_test_split,
)
from metrics_utils import evaluate_scores, select_threshold


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "model_ready_scaled.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "lr_naive_bayes_outputs"
TARGET_COLUMN = "survive_after_5"
RANDOM_STATE = 42


class LogisticRegressionScratch:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-values))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        self.weights = np.zeros(X.shape[1], dtype=np.float64)
        self.bias = 0.0
        for _ in range(self.epochs):
            probabilities = self._sigmoid(X @ self.weights + self.bias)
            errors = probabilities - y
            self.weights -= self.learning_rate * (X.T @ errors) / len(X)
            self.bias -= self.learning_rate * float(np.mean(errors))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self._sigmoid(X @ self.weights + self.bias)


class GaussianNaiveBayesScratch:
    def __init__(self, variance_smoothing: float = 1e-9):
        self.variance_smoothing = variance_smoothing
        self.classes: np.ndarray | None = None
        self.priors: dict[int, float] = {}
        self.means: dict[int, np.ndarray] = {}
        self.variances: dict[int, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayesScratch":
        self.classes = np.unique(y)
        for label in self.classes:
            class_rows = X[y == label]
            self.priors[int(label)] = len(class_rows) / len(X)
            self.means[int(label)] = class_rows.mean(axis=0)
            self.variances[int(label)] = (
                class_rows.var(axis=0) + self.variance_smoothing
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes is None:
            raise RuntimeError("Model must be fitted before prediction.")

        log_posteriors = []
        for label in self.classes:
            label = int(label)
            mean = self.means[label]
            variance = self.variances[label]
            log_density = (
                -0.5 * np.log(2 * np.pi * variance)
                - ((X - mean) ** 2) / (2 * variance)
            ).sum(axis=1)
            log_posteriors.append(np.log(self.priors[label]) + log_density)

        scores = np.column_stack(log_posteriors)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        class_1_index = int(np.where(self.classes == 1)[0][0])
        return probabilities[:, class_1_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")
    y = frame.pop(TARGET_COLUMN).astype(np.int8)
    X = frame.select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf], np.nan
    )
    return X.fillna(X.median(numeric_only=True)), y


def main() -> None:
    args = parse_args()
    X, y = load_data(args.data.resolve())
    X, y = stratified_sample(X, y, args.sample_size, RANDOM_STATE)
    split = stratified_train_validation_test_split(X, y, RANDOM_STATE)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegressionScratch(epochs=args.epochs),
        "naive_bayes": GaussianNaiveBayesScratch(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train.to_numpy())
        validation_scores = model.predict_proba(X_validation_scaled)
        threshold, threshold_frame = select_threshold(
            y_validation, validation_scores
        )
        test_scores = model.predict_proba(X_test_scaled)
        metrics, predictions = evaluate_scores(y_test, test_scores, threshold)
        rows.append({"model": name, **metrics})

        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        threshold_frame.to_csv(
            model_dir / "validation_threshold_metrics.csv", index=False
        )
        pd.DataFrame(
            {
                "actual": y_test.to_numpy(),
                "score_class_1": test_scores,
                "predicted": predictions,
            }
        ).to_csv(model_dir / "test_predictions.csv", index=False)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
