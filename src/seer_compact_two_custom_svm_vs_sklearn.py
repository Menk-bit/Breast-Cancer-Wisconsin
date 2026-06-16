
"""
SEER SVM comparison.

Models:
1. sklearn LinearSVC
2. Linear Mini-Batch SGD-SVM
3. RBF-Sampler Mini-Batch SGD-SVM
4. Ensemble of models 2 and 3

Input CSV must already be numeric, clean, and scaled.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import (
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

from data_splitting import (
    stratified_sample,
    stratified_train_validation_test_split,
    stratified_two_way_split,
)
from metrics_utils import (
    binary_classification_metrics,
    maximum_f1_threshold,
    roc_auc,
)
from result_reporting import save_explainability_artifacts, save_result_graphs

warnings.filterwarnings("ignore")


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = str(REPO_ROOT / "data" / "model_ready_scaled.csv")
OUTPUT_DIR = REPO_ROOT / "seer_two_custom_svm_outputs"
TARGET_COLUMN = "survive_after_5"
EXCLUDE_COLUMNS: list[str] = []

# Split
RANDOM_STATE = 42
META_TRAIN_FRACTION = 0.10

# Search
SEARCH_METHOD = "random" 
RUN_SEARCH = True
SEARCH_MAX_SAMPLES = 40_000
SEARCH_VALIDATION_SIZE = 0.20
SEARCH_RANDOM_STATE = 123

SKLEARN_SEARCH_N_ITER = 6
LINEAR_SEARCH_N_ITER = 10
RBF_SEARCH_N_ITER = 12
STACK_SEARCH_N_ITER = 8

SKLEARN_PARAM_GRID = {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]}
LINEAR_PARAM_GRID = {"regularization": [1e-5, 5e-5, 1e-4, 5e-4], "initial_lr": [0.02, 0.03, 0.05, 0.08], "epochs": [8, 12, 15, 20]}
RBF_PARAM_GRID = {"gamma": [0.005, 0.01, 0.02, 0.05], "n_components": [512, 1024, 2048], "regularization": [1e-5, 5e-5, 1e-4], "initial_lr": [0.02, 0.04, 0.06], "epochs": [6, 10, 15]}
STACK_PARAM_GRID = {"regularization": [1e-4, 5e-4, 1e-3, 5e-3], "initial_lr": [0.03, 0.05, 0.08, 0.12], "epochs": [40, 60, 80, 120], "balanced": [False, True]}

# Defaults
POSITIVE_WEIGHT_MULTIPLIER = 1.0

SKLEARN_C = 1.0
SKLEARN_TRAIN_STAGES = (0.25, 0.50, 0.75, 1.00)

LINEAR_REG = 5e-5
LINEAR_LR = 0.05
LINEAR_EPOCHS = 15
LINEAR_BATCH_SIZE = 2048

RBF_GAMMA = 0.02
RBF_COMPONENTS = 1024
RBF_REG = 5e-5
RBF_LR = 0.04
RBF_EPOCHS = 10
RBF_BATCH_SIZE = 2048
RBF_CHUNK_SIZE = 8192
RBF_PRECOMPUTE = True

STACK_REG = 1e-3
STACK_LR = 0.08
STACK_EPOCHS = 80
STACK_BATCH_SIZE = 1024
STACK_BALANCED = False

SHOW_PLOTS = False
PRINT_UPDATES = False
PRINT_EVERY = 50

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

def signed_labels(y: np.ndarray, positive_class: Any) -> np.ndarray:
    return np.where(y == positive_class, 1.0, -1.0).astype(np.float32)

def class_weight_map(y: np.ndarray, classes: np.ndarray, positive_multiplier: float) -> dict[Any, float]:
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    result = {label: float(weight) for label, weight in zip(classes, weights)}
    result[classes[1]] *= positive_multiplier
    return result

def sample_weights(y: np.ndarray, classes: np.ndarray, positive_multiplier: float, balanced: bool = True) -> np.ndarray:
    if not balanced:
        result = np.ones(len(y), dtype=np.float32)
        result[y == classes[1]] *= positive_multiplier
        return result
    mapping = class_weight_map(y, classes, positive_multiplier)
    return np.asarray([mapping[label] for label in y], dtype=np.float32)

def hinge_loss(y_signed: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    hinge = np.maximum(0.0, 1.0 - y_signed * scores)
    loss = float(np.dot(weights, hinge) / max(float(weights.sum()), 1e-12))
    return loss, float(np.mean(hinge > 0.0))

def primal_loss(X: np.ndarray, y_signed: np.ndarray, weights: np.ndarray, coef: np.ndarray, intercept: float, regularization: float) -> tuple[float, float, float, float]:
    hinge, violation = hinge_loss(y_signed, X @ coef + intercept, weights)
    reg = 0.5 * regularization * float(coef @ coef)
    return hinge + reg, hinge, reg, violation

def binary_cross_entropy(y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    losses = -(y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities))
    return float(np.dot(weights, losses) / max(float(weights.sum()), 1e-12))

def stratified_indices(y: np.ndarray, max_samples: int, random_state: int) -> np.ndarray:
    if len(y) <= max_samples:
        return np.arange(len(y))
    indices = np.arange(len(y))
    selected, _ = stratified_sample(indices, y, max_samples, random_state)
    return np.asarray(selected)

def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high - low <= 1e-12 else (values - low) / (high - low)

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")

def encode_target(target: pd.Series) -> np.ndarray:
    values = sorted(target.dropna().unique().tolist())
    if len(values) != 2:
        raise ValueError(f"Target must be binary. Found: {values}")
    if pd.api.types.is_numeric_dtype(target) and set(values) == {0, 1}:
        return target.astype(int).to_numpy()
    mapping = {values[0]: 0, values[1]: 1}
    print(f"  Target mapping: {mapping}")
    return target.map(mapping).astype(int).to_numpy()

def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    print("\n" + "=" * 88)
    print("STEP 1 - LOAD DATA")
    print("=" * 88)

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target '{TARGET_COLUMN}' not found.")

    df = df.loc[df[TARGET_COLUMN].notna()].copy()
    y = encode_target(df[TARGET_COLUMN])
    drop_columns = [TARGET_COLUMN, *[column for column in EXCLUDE_COLUMNS if column in df.columns]]
    features = df.drop(columns=drop_columns)

    non_numeric = features.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if non_numeric:
        raise TypeError(f"Non-numeric columns found: {non_numeric}")

    X = features.to_numpy(dtype=np.float32)
    if np.isnan(X).any() or not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity.")

    print(f"  Shape: {X.shape}")
    print(f"  Target: {TARGET_COLUMN}")
    print(f"  Distribution: {dict(pd.Series(y).value_counts().sort_index())}")
    print("  No preprocessing was applied.")
    return X, y, features.columns.tolist()

def split_roles(X: np.ndarray, y: np.ndarray):
    split = stratified_train_validation_test_split(X, y, RANDOM_STATE)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split
    X_base, X_meta, y_base, y_meta = stratified_two_way_split(
        X_train,
        y_train,
        test_size=META_TRAIN_FRACTION,
        random_state=RANDOM_STATE + 2,
    )
    return (
        X_base,
        X_meta,
        X_validation,
        X_test,
        y_base,
        y_meta,
        y_validation,
        y_test,
    )

def search_candidates(grid: dict[str, list[Any]], n_iter: int, random_state: int) -> list[dict[str, Any]]:
    if SEARCH_METHOD == "grid":
        return list(ParameterGrid(grid))
    if SEARCH_METHOD == "random":
        return list(ParameterSampler(grid, n_iter=min(n_iter, len(list(ParameterGrid(grid)))), random_state=random_state))
    raise ValueError("SEARCH_METHOD must be 'grid' or 'random'.")

def search_metrics(y_true: np.ndarray, scores: np.ndarray, positive_class: Any) -> dict[str, float]:
    threshold = maximum_f1_threshold(y_true, scores, positive_class)
    return {
        "precision": float(threshold["precision"]),
        "recall": float(threshold["recall"]),
        "f1": float(threshold["f1"]),
        "roc_auc": float(threshold["roc_auc"]),
    }

def better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    return (
        round(float(candidate["f1"]), 10),
        round(float(candidate["roc_auc"]), 10),
    ) > (
        round(float(best["f1"]), 10),
        round(float(best["roc_auc"]), 10),
    )

class SklearnLinearSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C: float, train_stages: tuple[float, ...], positive_multiplier: float, random_state: int):
        self.C = C
        self.train_stages = train_stages
        self.positive_multiplier = positive_multiplier
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, monitor_X: np.ndarray, monitor_y: np.ndarray):
        start_total = time.time()
        self.classes_ = np.unique(y)
        monitor_signed = signed_labels(monitor_y, self.classes_[1])
        monitor_weights = sample_weights(monitor_y, self.classes_, self.positive_multiplier)
        weights = class_weight_map(y, self.classes_, self.positive_multiplier)
        stages = sorted({min(len(X), max(100, int(len(X) * stage))) for stage in self.train_stages})
        if stages[-1] != len(X):
            stages.append(len(X))

        self.loss_history_ = []
        print("\n  Training sklearn LinearSVC...")

        for update, size in enumerate(stages, start=1):
            indices = stratified_indices(y, size, self.random_state + update)
            model = LinearSVC(C=self.C, class_weight=weights, dual="auto", max_iter=5000, tol=1e-4, random_state=self.random_state)
            stage_start = time.time()
            model.fit(X[indices], y[indices])
            loss, violation = hinge_loss(monitor_signed, model.decision_function(monitor_X), monitor_weights)
            self.loss_history_.append({"update": update, "train_size": len(indices), "monitor_hinge_loss": loss, "violation_rate": violation, "fit_seconds": time.time() - stage_start})
            self.model_ = model

        self.loss_history_df_ = pd.DataFrame(self.loss_history_)
        self.fit_time_ = time.time() - start_total
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.decision_function(X)

class LinearMiniBatchSGDSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, regularization: float, initial_lr: float, epochs: int, batch_size: int, positive_multiplier: float, random_state: int):
        self.regularization = regularization
        self.initial_lr = initial_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.positive_multiplier = positive_multiplier
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        start_total = time.time()
        self.classes_ = np.unique(y)
        y_signed = signed_labels(y, self.classes_[1])
        weights = sample_weights(y, self.classes_, self.positive_multiplier)
        self.coef_ = np.zeros(X.shape[1], dtype=np.float32)
        self.intercept_ = 0.0
        self.loss_history_ = []
        rng = np.random.default_rng(self.random_state)
        update = 0

        print("\n  Training Linear Mini-Batch SGD-SVM...")

        for epoch in range(1, self.epochs + 1):
            shuffled = rng.permutation(len(X))
            for batch, start in enumerate(range(0, len(X), self.batch_size), start=1):
                update += 1
                indices = shuffled[start:start + self.batch_size]
                X_batch, y_batch, w_batch = X[indices], y_signed[indices], weights[indices]
                scores = X_batch @ self.coef_ + self.intercept_
                violating = y_batch * scores < 1.0
                denominator = max(float(w_batch.sum()), 1e-12)
                grad_w = self.regularization * self.coef_
                grad_b = 0.0

                if np.any(violating):
                    effective = w_batch[violating] * y_batch[violating]
                    grad_w -= X_batch[violating].T @ effective / denominator
                    grad_b -= float(effective.sum() / denominator)

                lr = self.initial_lr / math.sqrt(update)
                self.coef_ -= lr * grad_w.astype(np.float32)
                self.intercept_ -= lr * grad_b
                total, hinge, reg, violation = primal_loss(X_batch, y_batch, w_batch, self.coef_, self.intercept_, self.regularization)
                self.loss_history_.append({"update": update, "epoch": epoch, "batch": batch, "learning_rate": lr, "hinge_loss": hinge, "regularization_loss": reg, "total_loss": total, "violation_rate": violation})

                if PRINT_UPDATES and update % PRINT_EVERY == 0:
                    print(f"    update={update} loss={total:.6f}")

        self.loss_history_df_ = pd.DataFrame(self.loss_history_)
        self.fit_time_ = time.time() - start_total
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return X @ self.coef_ + self.intercept_

class RBFSamplerSGDSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma: float, n_components: int, regularization: float, initial_lr: float, epochs: int, batch_size: int, chunk_size: int, precompute: bool, positive_multiplier: float, random_state: int):
        self.gamma = gamma
        self.n_components = n_components
        self.regularization = regularization
        self.initial_lr = initial_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.precompute = precompute
        self.positive_multiplier = positive_multiplier
        self.random_state = random_state

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = np.empty((len(X), self.n_components), dtype=np.float32)
        for start in range(0, len(X), self.chunk_size):
            end = min(start + self.chunk_size, len(X))
            result[start:end] = self.mapper_.transform(X[start:end]).astype(np.float32)
        return result

    def fit(self, X: np.ndarray, y: np.ndarray):
        start_total = time.time()
        self.classes_ = np.unique(y)
        y_signed = signed_labels(y, self.classes_[1])
        weights = sample_weights(y, self.classes_, self.positive_multiplier)
        self.mapper_ = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=self.random_state)
        self.mapper_.fit(X[:1])
        transformed = self.transform(X) if self.precompute else None
        self.coef_ = np.zeros(self.n_components, dtype=np.float32)
        self.intercept_ = 0.0
        self.loss_history_ = []
        rng = np.random.default_rng(self.random_state)
        update = 0

        print("\n  Training RBF-Sampler SGD-SVM...")

        for epoch in range(1, self.epochs + 1):
            shuffled = rng.permutation(len(X))
            for batch, start in enumerate(range(0, len(X), self.batch_size), start=1):
                update += 1
                indices = shuffled[start:start + self.batch_size]
                Z_batch = transformed[indices] if transformed is not None else self.mapper_.transform(X[indices]).astype(np.float32)
                y_batch, w_batch = y_signed[indices], weights[indices]
                scores = Z_batch @ self.coef_ + self.intercept_
                violating = y_batch * scores < 1.0
                denominator = max(float(w_batch.sum()), 1e-12)
                grad_w = self.regularization * self.coef_
                grad_b = 0.0

                if np.any(violating):
                    effective = w_batch[violating] * y_batch[violating]
                    grad_w -= Z_batch[violating].T @ effective / denominator
                    grad_b -= float(effective.sum() / denominator)

                lr = self.initial_lr / math.sqrt(update)
                self.coef_ -= lr * grad_w.astype(np.float32)
                self.intercept_ -= lr * grad_b
                total, hinge, reg, violation = primal_loss(Z_batch, y_batch, w_batch, self.coef_, self.intercept_, self.regularization)
                self.loss_history_.append({"update": update, "epoch": epoch, "batch": batch, "learning_rate": lr, "hinge_loss": hinge, "regularization_loss": reg, "total_loss": total, "violation_rate": violation})

                if PRINT_UPDATES and update % PRINT_EVERY == 0:
                    print(f"    update={update} loss={total:.6f}")

        self.loss_history_df_ = pd.DataFrame(self.loss_history_)
        self.fit_time_ = time.time() - start_total
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        scores = np.empty(len(X), dtype=np.float32)
        for start in range(0, len(X), self.chunk_size):
            end = min(start + self.chunk_size, len(X))
            Z = self.mapper_.transform(X[start:end]).astype(np.float32)
            scores[start:end] = Z @ self.coef_ + self.intercept_
        return scores

class LogisticStacker(BaseEstimator, ClassifierMixin):
    def __init__(self, regularization: float, initial_lr: float, epochs: int, batch_size: int, balanced: bool, random_state: int):
        self.regularization = regularization
        self.initial_lr = initial_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.balanced = balanced
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        start_total = time.time()
        self.classes_ = np.unique(y)
        y_binary = (y == self.classes_[1]).astype(np.float32)
        weights = sample_weights(y, self.classes_, 1.0, balanced=self.balanced)
        self.coef_ = np.zeros(X.shape[1], dtype=np.float32)
        self.intercept_ = 0.0
        self.loss_history_ = []
        rng = np.random.default_rng(self.random_state)
        update = 0

        print("\n  Training ensemble stacker...")

        for epoch in range(1, self.epochs + 1):
            shuffled = rng.permutation(len(X))
            for batch, start in enumerate(range(0, len(X), self.batch_size), start=1):
                update += 1
                indices = shuffled[start:start + self.batch_size]
                X_batch, y_batch, w_batch = X[indices], y_binary[indices], weights[indices]
                probabilities = sigmoid(X_batch @ self.coef_ + self.intercept_)
                denominator = max(float(w_batch.sum()), 1e-12)
                error = (probabilities - y_batch) * w_batch
                grad_w = X_batch.T @ error / denominator + self.regularization * self.coef_
                grad_b = float(error.sum() / denominator)
                lr = self.initial_lr / math.sqrt(update)
                self.coef_ -= lr * grad_w.astype(np.float32)
                self.intercept_ -= lr * grad_b
                updated = sigmoid(X_batch @ self.coef_ + self.intercept_)
                bce = binary_cross_entropy(y_batch, updated, w_batch)
                reg = 0.5 * self.regularization * float(self.coef_ @ self.coef_)
                self.loss_history_.append({"update": update, "epoch": epoch, "batch": batch, "learning_rate": lr, "binary_cross_entropy": bce, "regularization_loss": reg, "total_loss": bce + reg})

        self.loss_history_df_ = pd.DataFrame(self.loss_history_)
        self.fit_time_ = time.time() - start_total
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return X @ self.coef_ + self.intercept_

class TwoModelEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, linear_model: Any, rbf_model: Any, random_state: int):
        self.linear_model = linear_model
        self.rbf_model = rbf_model
        self.random_state = random_state

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([self.linear_model.decision_function(X), self.rbf_model.decision_function(X)]).astype(np.float32)

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray, X_threshold: np.ndarray, y_threshold: np.ndarray):
        self.classes_ = np.unique(y_meta)
        meta_scores = self.score_matrix(X_meta)
        self.stacker_, self.scaler_, self.search_results_, self.best_params_ = tune_stacker(meta_scores, y_meta, self.random_state)
        threshold_scores = self.scaler_.transform(self.score_matrix(X_threshold)).astype(np.float32)
        threshold_result = maximum_f1_threshold(
            y_threshold,
            self.stacker_.decision_function(threshold_scores),
            self.classes_[1],
        )
        self.threshold_ = threshold_result["threshold"]
        self.threshold_precision_ = threshold_result["precision"]
        self.threshold_recall_ = threshold_result["recall"]
        self.threshold_f1_ = threshold_result["f1"]
        self.threshold_curve_ = threshold_result["curve"]
        self.loss_history_df_ = self.stacker_.loss_history_df_.copy()
        self.fit_time_ = self.linear_model.fit_time_ + self.rbf_model.fit_time_ + self.stacker_.fit_time_
        self.coefficients_ = pd.DataFrame({"model": ["Linear Mini-Batch SGD-SVM", "RBF-Sampler SGD-SVM"], "coefficient": self.stacker_.coef_})
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores = self.scaler_.transform(self.score_matrix(X)).astype(np.float32)
        return self.stacker_.decision_function(scores)

def tune_stacker(meta_scores: np.ndarray, y_meta: np.ndarray, random_state: int):
    X_train, X_valid, y_train, y_valid = stratified_two_way_split(
        meta_scores,
        y_meta,
        test_size=0.25,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_valid_scaled = scaler.transform(X_valid).astype(np.float32)
    candidates = search_candidates(STACK_PARAM_GRID, STACK_SEARCH_N_ITER, random_state)
    best_row = None
    rows = []

    for index, params in enumerate(candidates, start=1):
        model = LogisticStacker(regularization=float(params["regularization"]), initial_lr=float(params["initial_lr"]), epochs=int(params["epochs"]), batch_size=STACK_BATCH_SIZE, balanced=bool(params["balanced"]), random_state=random_state + index)
        model.fit(X_train_scaled, y_train)
        metrics = search_metrics(y_valid, model.decision_function(X_valid_scaled), np.unique(y_train)[1])
        row = {"candidate": index, **params, **metrics}
        rows.append(row)
        if better(row, best_row):
            best_row = row

    assert best_row is not None
    final_scaler = StandardScaler()
    full_scaled = final_scaler.fit_transform(meta_scores).astype(np.float32)
    final_model = LogisticStacker(regularization=float(best_row["regularization"]), initial_lr=float(best_row["initial_lr"]), epochs=int(best_row["epochs"]), batch_size=STACK_BATCH_SIZE, balanced=bool(best_row["balanced"]), random_state=random_state)
    final_model.fit(full_scaled, y_meta)
    return final_model, final_scaler, pd.DataFrame(rows), best_row

def run_search(X_base: np.ndarray, y_base: np.ndarray, output_dir: Path) -> dict[str, Any]:
    global SKLEARN_C, LINEAR_REG, LINEAR_LR, LINEAR_EPOCHS, RBF_GAMMA, RBF_COMPONENTS, RBF_REG, RBF_LR, RBF_EPOCHS

    output_dir.mkdir(parents=True, exist_ok=True)
    indices = stratified_indices(y_base, min(SEARCH_MAX_SAMPLES, len(y_base)), SEARCH_RANDOM_STATE)
    X_sample, y_sample = X_base[indices], y_base[indices]
    X_train, X_valid, y_train, y_valid = stratified_two_way_split(
        X_sample,
        y_sample,
        test_size=SEARCH_VALIDATION_SIZE,
        random_state=SEARCH_RANDOM_STATE,
    )
    positive_class = np.unique(y_train)[1]
    rows = []

    print("\n" + "=" * 88)
    print("STEP 3 - HYPERPARAMETER SEARCH")
    print("=" * 88)

    best_sklearn = None
    for index, params in enumerate(search_candidates(SKLEARN_PARAM_GRID, SKLEARN_SEARCH_N_ITER, SEARCH_RANDOM_STATE), start=1):
        model = LinearSVC(C=float(params["C"]), class_weight=class_weight_map(y_train, np.unique(y_train), POSITIVE_WEIGHT_MULTIPLIER), dual="auto", max_iter=5000, tol=1e-4, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        metrics = search_metrics(y_valid, model.decision_function(X_valid), positive_class)
        row = {"model": "sklearn LinearSVC", "candidate": index, **params, **metrics}
        rows.append(row)
        if better(row, best_sklearn):
            best_sklearn = row

    best_linear = None
    for index, params in enumerate(search_candidates(LINEAR_PARAM_GRID, LINEAR_SEARCH_N_ITER, SEARCH_RANDOM_STATE + 1), start=1):
        model = LinearMiniBatchSGDSVM(regularization=float(params["regularization"]), initial_lr=float(params["initial_lr"]), epochs=int(params["epochs"]), batch_size=LINEAR_BATCH_SIZE, positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER, random_state=RANDOM_STATE + index)
        model.fit(X_train, y_train)
        metrics = search_metrics(y_valid, model.decision_function(X_valid), positive_class)
        row = {"model": "Linear Mini-Batch SGD-SVM", "candidate": index, **params, **metrics}
        rows.append(row)
        if better(row, best_linear):
            best_linear = row

    best_rbf = None
    for index, params in enumerate(search_candidates(RBF_PARAM_GRID, RBF_SEARCH_N_ITER, SEARCH_RANDOM_STATE + 2), start=1):
        model = RBFSamplerSGDSVM(gamma=float(params["gamma"]), n_components=int(params["n_components"]), regularization=float(params["regularization"]), initial_lr=float(params["initial_lr"]), epochs=int(params["epochs"]), batch_size=RBF_BATCH_SIZE, chunk_size=RBF_CHUNK_SIZE, precompute=True, positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER, random_state=RANDOM_STATE + 100 + index)
        model.fit(X_train, y_train)
        metrics = search_metrics(y_valid, model.decision_function(X_valid), positive_class)
        row = {"model": "RBF-Sampler SGD-SVM", "candidate": index, **params, **metrics}
        rows.append(row)
        if better(row, best_rbf):
            best_rbf = row

    assert best_sklearn and best_linear and best_rbf

    SKLEARN_C = float(best_sklearn["C"])
    LINEAR_REG = float(best_linear["regularization"])
    LINEAR_LR = float(best_linear["initial_lr"])
    LINEAR_EPOCHS = int(best_linear["epochs"])
    RBF_GAMMA = float(best_rbf["gamma"])
    RBF_COMPONENTS = int(best_rbf["n_components"])
    RBF_REG = float(best_rbf["regularization"])
    RBF_LR = float(best_rbf["initial_lr"])
    RBF_EPOCHS = int(best_rbf["epochs"])

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "search_results.csv", index=False)
    pd.DataFrame([best_sklearn, best_linear, best_rbf]).to_csv(output_dir / "selected_parameters.csv", index=False)

    print(f"  sklearn LinearSVC: C={SKLEARN_C}")
    print(f"  Linear SGD: reg={LINEAR_REG}, lr={LINEAR_LR}, epochs={LINEAR_EPOCHS}")
    print(f"  RBF SGD: gamma={RBF_GAMMA}, components={RBF_COMPONENTS}, reg={RBF_REG}, lr={RBF_LR}, epochs={RBF_EPOCHS}")
    return {"results": results, "selected": pd.DataFrame([best_sklearn, best_linear, best_rbf])}

def save_line(frame: pd.DataFrame, x: str, columns: list[str], title: str, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for column in columns:
        if column in frame.columns:
            plt.plot(frame[x], frame[column], label=column)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    if len(columns) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def evaluate(
    name: str,
    model: Any,
    X_threshold: np.ndarray,
    y_threshold: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    loss_x: str,
    loss_columns: list[str],
    X_background: np.ndarray,
    feature_names: list[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_class = np.unique(y_threshold)[1]
    negative_class = np.unique(y_threshold)[0]

    if hasattr(model, "threshold_"):
        threshold = model.threshold_
        validation_precision = model.threshold_precision_
        validation_recall = model.threshold_recall_
        validation_f1 = model.threshold_f1_
        threshold_curve = model.threshold_curve_
    else:
        result = maximum_f1_threshold(
            y_threshold,
            model.decision_function(X_threshold),
            positive_class,
        )
        threshold = result["threshold"]
        validation_precision = result["precision"]
        validation_recall = result["recall"]
        validation_f1 = result["f1"]
        threshold_curve = result["curve"]

    scores = model.decision_function(X_test)
    predictions = np.where(scores >= threshold, positive_class, negative_class)
    metrics = {
        "experiment": name,
        "threshold": threshold,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "validation_f1": validation_f1,
        **binary_classification_metrics(y_test, predictions, scores),
        "training_seconds": float(model.fit_time_),
    }

    print("\n" + "-" * 88)
    print(name)
    print("-" * 88)
    print(f"  Validation P/R/F1: {validation_precision:.4f} / {validation_recall:.4f} / {validation_f1:.4f}")
    print(f"  Test class-1 F1: {metrics['f1_class_1']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    threshold_curve.to_csv(output_dir / "threshold_metrics.csv", index=False)
    model.loss_history_df_.to_csv(output_dir / "loss_history.csv", index=False)
    save_line(
        threshold_curve,
        "threshold",
        ["precision_class_1", "recall_class_1", "f1_class_1"],
        f"{name}: threshold metrics",
        output_dir / "threshold_curve.png",
    )
    save_line(model.loss_history_df_, loss_x, loss_columns, f"{name}: loss", output_dir / "loss_curve.png")
    save_result_graphs(y_test, scores, predictions, metrics, output_dir, name)

    def predict_proba(rows):
        class_1 = sigmoid(model.decision_function(np.asarray(rows, dtype=np.float32)))
        return np.column_stack([1.0 - class_1, class_1])

    save_explainability_artifacts(
        model_name=name,
        output_dir=output_dir / "explainability",
        X_background=X_background,
        X_explain=X_test,
        feature_names=feature_names,
        predict_proba_fn=predict_proba,
    )

    if hasattr(model, "coefficients_"):
        model.coefficients_.to_csv(output_dir / "ensemble_coefficients.csv", index=False)
        model.search_results_.to_csv(output_dir / "stacker_search_results.csv", index=False)
        pd.DataFrame([model.best_params_]).to_csv(output_dir / "selected_stacker_parameters.csv", index=False)

    return metrics, {"scores": scores, "loss_history": model.loss_history_df_.copy()}

def export_comparison(rows: list[dict[str, Any]], details: dict[str, dict[str, Any]], y_test: np.ndarray, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["rank_validation_f1"] = frame["validation_f1"].rank(ascending=False, method="min").astype(int)
    frame["rank_test_f1"] = frame["f1_class_1"].rank(ascending=False, method="min").astype(int)
    frame = frame.sort_values(["rank_validation_f1", "rank_test_f1"])
    frame.to_csv(output_dir / "comparison.csv", index=False)

    x = np.arange(len(frame))
    width = 0.16
    plt.figure(figsize=(12, 7))
    for offset, metric in enumerate(
        [
            "accuracy",
            "precision_class_0",
            "precision_class_1",
            "f1_class_0",
            "f1_class_1",
        ]
    ):
        plt.bar(x + (offset - 2) * width, frame[metric], width=width, label=metric)
    plt.xticks(x, frame["experiment"], rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 7))
    for name, item in details.items():
        fpr, tpr, _ = roc_curve(y_test, item["scores"])
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc(y_test, item['scores']):.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "all_roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, item in details.items():
        history = item["loss_history"]
        column = "total_loss" if "total_loss" in history.columns else "monitor_hinge_loss"
        values = history[column].to_numpy(dtype=float)
        plt.plot(np.linspace(0, 1, len(values)), normalize(values), label=name)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "normalized_loss_curves.png", dpi=200)
    plt.close()

    return frame

def main(data_path: str = DATA_PATH, output_dir: str | Path = OUTPUT_DIR, quick_test: bool = False):
    global RUN_SEARCH, SKLEARN_TRAIN_STAGES, LINEAR_EPOCHS, RBF_COMPONENTS, RBF_EPOCHS, STACK_EPOCHS

    if quick_test:
        RUN_SEARCH = False
        SKLEARN_TRAIN_STAGES = (0.5, 1.0)
        LINEAR_EPOCHS = 3
        RBF_COMPONENTS = 128
        RBF_EPOCHS = 3
        STACK_EPOCHS = 10

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    X, y, feature_names = load_data(data_path)
    (
        X_base,
        X_meta,
        X_validation,
        X_test,
        y_base,
        y_meta,
        y_validation,
        y_test,
    ) = split_roles(X, y)

    print("\n" + "=" * 88)
    print("STEP 2 - SPLIT")
    print("=" * 88)
    print(f"  Base: {X_base.shape}")
    print(f"  Meta: {X_meta.shape}")
    print(f"  Training total: {len(X_base) + len(X_meta):,} rows (60%)")
    print(f"  Validation: {X_validation.shape} (20%)")
    print(f"  Test: {X_test.shape}")

    search_artifacts = run_search(X_base, y_base, output_dir / "search") if RUN_SEARCH and not quick_test else None

    print("\n" + "=" * 88)
    print("STEP 4 - TRAIN FINAL MODELS")
    print("=" * 88)

    sklearn_model = SklearnLinearSVC(C=SKLEARN_C, train_stages=SKLEARN_TRAIN_STAGES, positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER, random_state=RANDOM_STATE)
    sklearn_model.fit(X_base, y_base, X_validation, y_validation)

    linear_model = LinearMiniBatchSGDSVM(regularization=LINEAR_REG, initial_lr=LINEAR_LR, epochs=LINEAR_EPOCHS, batch_size=LINEAR_BATCH_SIZE, positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER, random_state=RANDOM_STATE)
    linear_model.fit(X_base, y_base)

    rbf_model = RBFSamplerSGDSVM(gamma=RBF_GAMMA, n_components=RBF_COMPONENTS, regularization=RBF_REG, initial_lr=RBF_LR, epochs=RBF_EPOCHS, batch_size=RBF_BATCH_SIZE, chunk_size=RBF_CHUNK_SIZE, precompute=RBF_PRECOMPUTE, positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER, random_state=RANDOM_STATE)
    rbf_model.fit(X_base, y_base)

    ensemble = TwoModelEnsemble(linear_model=linear_model, rbf_model=rbf_model, random_state=RANDOM_STATE + 100)
    ensemble.fit(X_meta, y_meta, X_validation, y_validation)

    experiments = [
        ("sklearn LinearSVC", sklearn_model, "train_size", ["monitor_hinge_loss"]),
        ("Linear Mini-Batch SGD-SVM", linear_model, "update", ["total_loss", "hinge_loss"]),
        ("RBF-Sampler SGD-SVM", rbf_model, "update", ["total_loss", "hinge_loss"]),
        ("Linear + RBF Ensemble", ensemble, "update", ["total_loss", "binary_cross_entropy"]),
    ]

    rows = []
    details = {}

    for name, model, loss_x, loss_columns in experiments:
        metrics, item = evaluate(
            name,
            model,
            X_validation,
            y_validation,
            X_test,
            y_test,
            output_dir / "experiments" / safe_name(name),
            loss_x,
            loss_columns,
            X_base,
            feature_names,
        )
        rows.append(metrics)
        details[name] = item

    comparison = export_comparison(rows, details, y_test, output_dir / "comparison")

    print("\n" + "=" * 88)
    print("FINAL COMPARISON")
    print("=" * 88)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n  Features: {len(feature_names)}")
    print(f"  Runtime: {(time.time() - total_start) / 60:.2f} minutes")
    print(f"  Outputs: {output_dir}")

    return {"models": {"sklearn": sklearn_model, "linear_sgd": linear_model, "rbf_sgd": rbf_model, "ensemble": ensemble}, "comparison": comparison, "details": details, "search": search_artifacts}

if __name__ == "__main__":
    main()
