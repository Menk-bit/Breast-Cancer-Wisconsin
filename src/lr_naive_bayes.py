"""From-scratch logistic regression and Gaussian Naive Bayes comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from data_splitting import (
    stratified_sample,
    stratified_train_validation_test_split,
)
from metrics_utils import evaluate_scores, select_threshold
from result_reporting import save_explainability_artifacts, save_result_graphs


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
    parser.add_argument(
        "--explainability-sample-size",
        type=int,
        default=200,
        help="Maximum rows used for SHAP and LIME background/explanation samples.",
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip SHAP and LIME artifact generation.",
    )
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


def model_grids(default_epochs: int) -> dict[str, list[dict[str, float | int]]]:
    return {
        "logistic_regression": list(
            ParameterGrid(
                {
                    "learning_rate": [0.001, 0.005, 0.01],
                    "epochs": [max(50, default_epochs // 2), default_epochs],
                }
            )
        ),
        "naive_bayes": list(
            ParameterGrid(
                {
                    "variance_smoothing": [1e-12, 1e-9, 1e-6, 1e-3],
                }
            )
        ),
    }


def build_model(name: str, params: dict[str, float | int]):
    if name == "logistic_regression":
        return LogisticRegressionScratch(
            learning_rate=float(params["learning_rate"]),
            epochs=int(params["epochs"]),
        )
    if name == "naive_bayes":
        return GaussianNaiveBayesScratch(
            variance_smoothing=float(params["variance_smoothing"])
        )
    raise ValueError(f"Unknown model: {name}")


def run_hyperparameter_search(
    name: str,
    grid: list[dict[str, float | int]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: pd.Series,
    output_dir: Path,
):
    rows = []
    best_model = None
    best_row = None
    best_threshold_frame = None

    for config_id, params in enumerate(grid, start=1):
        model = build_model(name, params)
        model.fit(X_train, y_train)
        validation_scores = model.predict_proba(X_validation)
        threshold, threshold_frame = select_threshold(y_validation, validation_scores)
        metrics, _ = evaluate_scores(y_validation, validation_scores, threshold)
        row = {
            "config_id": config_id,
            **params,
            **metrics,
        }
        rows.append(row)

        is_best = best_row is None or (
            row["f1_class_1"],
            row["roc_auc"],
            row["f1_class_0"],
        ) > (
            best_row["f1_class_1"],
            best_row["roc_auc"],
            best_row["f1_class_0"],
        )
        if is_best:
            best_model = model
            best_row = row
            best_threshold_frame = threshold_frame

        print(
            f"{name} config {config_id:02d}/{len(grid)} | "
            f"f1_1={metrics['f1_class_1']:.4f}, "
            f"f1_0={metrics['f1_class_0']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, threshold={threshold:.3f}"
        )

    if best_model is None or best_row is None or best_threshold_frame is None:
        raise RuntimeError(f"No hyperparameter configs were evaluated for {name}.")

    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    search_frame = pd.DataFrame(rows).sort_values(
        ["f1_class_1", "roc_auc", "f1_class_0"],
        ascending=False,
    )
    search_frame.to_csv(model_dir / "hyperparameter_search.csv", index=False)
    best_threshold_frame.to_csv(
        model_dir / "validation_threshold_metrics.csv", index=False
    )
    pd.Series(best_row).to_frame().T.to_csv(
        model_dir / "selected_hyperparameters.csv", index=False
    )
    plot_hyperparameter_curve(name, search_frame, model_dir)
    plot_threshold_curve(best_threshold_frame, model_dir)
    return best_model, pd.Series(best_row)


def plot_hyperparameter_curve(name: str, search_frame: pd.DataFrame, output_dir: Path):
    x_column = "learning_rate" if "learning_rate" in search_frame.columns else "variance_smoothing"
    grouped = (
        search_frame.groupby(x_column)[["f1_class_1", "f1_class_0", "roc_auc"]]
        .max()
        .reset_index()
        .sort_values(x_column)
    )
    plt.figure(figsize=(9, 5))
    for metric in ["f1_class_1", "f1_class_0", "roc_auc"]:
        plt.plot(grouped[x_column], grouped[metric], marker="o", label=metric)
    if x_column == "variance_smoothing":
        plt.xscale("log")
    plt.title(f"{name}: Hyperparameter Curve")
    plt.xlabel(x_column)
    plt.ylabel("Validation score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparameter_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_threshold_curve(threshold_frame: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(9, 5))
    for metric in ["f1_class_1", "f1_class_0", "roc_auc"]:
        plt.plot(threshold_frame["threshold"], threshold_frame[metric], label=metric)
    plt.title("Validation Threshold Curve")
    plt.xlabel("Threshold")
    plt.ylabel("Validation score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, grid in model_grids(args.epochs).items():
        print(f"\nSearching {name} hyperparameters...")
        model, best_row = run_hyperparameter_search(
            name,
            grid,
            X_train_scaled,
            y_train.to_numpy(),
            X_validation_scaled,
            y_validation,
            output_dir,
        )
        threshold = float(best_row["threshold"])
        test_scores = model.predict_proba(X_test_scaled)
        metrics, predictions = evaluate_scores(y_test, test_scores, threshold)
        rows.append({"model": name, **metrics})

        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "actual": y_test.to_numpy(),
                "score_class_1": test_scores,
                "predicted": predictions,
            }
        ).to_csv(model_dir / "test_predictions.csv", index=False)
        save_result_graphs(y_test, test_scores, predictions, metrics, model_dir, name)

        def predict_proba(rows, fitted_model=model):
            class_1 = fitted_model.predict_proba(np.asarray(rows, dtype=float))
            return np.column_stack([1.0 - class_1, class_1])

        if not args.skip_explainability:
            save_explainability_artifacts(
                model_name=name,
                output_dir=model_dir / "explainability",
                X_background=pd.DataFrame(X_train_scaled, columns=X.columns),
                X_explain=pd.DataFrame(X_test_scaled, columns=X.columns),
                feature_names=X.columns.tolist(),
                predict_proba_fn=predict_proba,
                shap_sample_size=args.explainability_sample_size,
                background_sample_size=args.explainability_sample_size,
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
