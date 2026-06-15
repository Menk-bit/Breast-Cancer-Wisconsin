"""
SEER Breast Cancer — CVM-RBF + Linear SGD-SVM + RBF-Sampler SGD-SVM
with logistic stacking, complete loss tracking, lightweight tuning, and maximum-F1 thresholding.

Pipeline
--------
1. Split untouched test data.
2. Split the remaining train data into:
   - base-training set: trains the three base learners;
   - meta-training set: trains the stacking model;
   - threshold-validation set: selects the final decision threshold.
3. Fit preprocessing, Mutual Information feature selection, and scaling only
   on the base-training set.
4. Train:
   - CVM-style RBF-SVC on progressively larger boundary-near core sets;
   - primal mini-batch linear SGD-SVM;
   - RBFSampler + primal mini-batch SGD-SVM.
5. Train a logistic SGD stacker on the three validation decision scores.
6. Select the final threshold that maximizes F1 on threshold-validation data.
7. Export loss histories, plots, diversity diagnostics, and test metrics.

Important terminology
---------------------
The CVM component is a practical CVM-style core-vector selection heuristic.
It is not the original Minimum Enclosing Ball CVM algorithm.
"""

from __future__ import annotations

import math
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

DATA_PATH = "./data/model_ready_tree.csv"
OUTPUT_DIR = Path("./seer_cvm_sgd_rbf_max_f1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
META_FRACTION = 0.10
THRESHOLD_FRACTION = 0.10

# Feature selection
TOP_K_FEATURES = 65
MI_MAX_SAMPLES = 50_000

# Potential leakage / follow-up-bias controls
DROP_SURVIVAL_DERIVED_FEATURES = True
DROP_DIAGNOSIS_TIME_FEATURES = False

# Shared class emphasis. Threshold tuning remains the primary recall control.
POSITIVE_WEIGHT_MULTIPLIER = 1.00

# Pilot model used only to rank distance from the linear decision boundary.
PILOT_C = 1.0

# CVM-style RBF model
CVM_CORE_STAGES = (500, 1_000, 2_000, 3_000)
CVM_C = 2.0
CVM_GAMMA: str | float = 0.02
CVM_CACHE_SIZE_MB = 2_048
CVM_BALANCED_CORE = True

# Linear primal SGD-SVM
LINEAR_SVM_LAMBDA = 5e-5
LINEAR_SVM_INITIAL_LR = 0.05
LINEAR_SVM_EPOCHS = 15
LINEAR_SVM_BATCH_SIZE = 2_048

# Approximate nonlinear RBF-SVM
RBF_GAMMA = 0.02
RBF_COMPONENTS = 1_024
RBF_SVM_LAMBDA = 5e-5
RBF_SVM_INITIAL_LR = 0.04
RBF_SVM_EPOCHS = 10
RBF_SVM_BATCH_SIZE = 2_048
RBF_TRANSFORM_CHUNK_SIZE = 8_192
RBF_PRECOMPUTE_TRAIN = True

# Logistic stacking model
STACK_LAMBDA = 1e-3
STACK_INITIAL_LR = 0.08
STACK_EPOCHS = 80
STACK_BATCH_SIZE = 1_024
STACK_USE_BALANCED_WEIGHTS = False

# Threshold optimization
THRESHOLD_MODE = "max_f1"  # direct maximum-F1 thresholding
TARGET_RECALL = 0.85
MIN_PRECISION = 0.58
THRESHOLD_BETA = 1.0
RECALL_SHORTFALL_PENALTY = 2.0
PRECISION_SHORTFALL_PENALTY = 1.0

# Logging / outputs
LOSS_MONITOR_SAMPLES = 10_000
PRINT_EVERY_UPDATE = False
PRINT_EVERY_N_UPDATES = 1
SHOW_PLOTS = False
SAVE_MODEL_ARTIFACTS = False


# Lightweight hyperparameter tuning
# Tuning is performed only on a stratified subset of the base-training role.
# The selected configuration is then retrained on the complete base-training set.
RUN_LIGHTWEIGHT_TUNING = True
TUNING_MAX_SAMPLES = 60_000
TUNING_VALIDATION_SIZE = 0.20
TUNING_RANDOM_STATE = 123

LINEAR_TUNING_CANDIDATES = [
    {"regularization": 1e-5, "epochs": 10, "initial_lr": 0.05},
    {"regularization": 5e-5, "epochs": 12, "initial_lr": 0.05},
    {"regularization": 1e-4, "epochs": 12, "initial_lr": 0.04},
    {"regularization": 5e-4, "epochs": 10, "initial_lr": 0.03},
]

RBF_TUNING_CANDIDATES = [
    {"gamma": 0.01, "n_components": 512,  "regularization": 5e-5, "epochs": 6},
    {"gamma": 0.02, "n_components": 512,  "regularization": 5e-5, "epochs": 6},
    {"gamma": 0.02, "n_components": 1024, "regularization": 5e-5, "epochs": 6},
    {"gamma": 0.05, "n_components": 1024, "regularization": 1e-4, "epochs": 6},
]

CVM_TUNING_CANDIDATES = [
    {"C": 1.0, "gamma": 0.01},
    {"C": 2.0, "gamma": 0.02},
    {"C": 5.0, "gamma": 0.02},
]
CVM_TUNING_CORE_STAGES = (500, 1_500)


# ============================================================================
# DATA COLUMNS
# ============================================================================

SEER_TARGET_CANDIDATES = [
    "survive_after_5",
    "Dead",
    "Status",
    "Vital status recode (study cutoff used)",
    "10-year survival",
    "cause_of_death",
]

BASE_DROP_COLUMNS = [
    "Patient ID",
    "patient_id",
    "Case Number",
    "survival_months_int",
    "Survival months",
]

SURVIVAL_DERIVED_COLUMNS = [
    "survival_months_unknown_flag",
]

DIAGNOSIS_TIME_COLUMNS = [
    "diagnosis_year",
    "diagnosis_era_2018_2022",
    "diagnosis_era_2023_plus",
    "diagnosis_era_pre_2010",
]


# ============================================================================
# LOSS DEFINITIONS AND NUMERICAL HELPERS
# ============================================================================

def print_loss_definitions() -> None:
    print("\n" + "=" * 84)
    print("LOSS FUNCTIONS")
    print("=" * 84)
    print(
        "CVM-RBF monitoring loss after each core-set update:\n"
        "  L_CVM = sum_i c_i max(0, 1 - y_i f_RBF(x_i)) / sum_i c_i\n"
        "  Note: sklearn SVC does not expose LIBSVM's internal dual objective,\n"
        "  so this is a fixed-monitor weighted hinge loss."
    )
    print(
        "Linear Mini-Batch SGD-SVM:\n"
        "  J_linear = (lambda/2)||w||^2 + "
        "sum_i c_i max(0, 1-y_i(w^T x_i+b))/sum_i c_i"
    )
    print(
        "RBF-Sampler Mini-Batch SGD-SVM:\n"
        "  J_rbf = (lambda/2)||v||^2 + "
        "sum_i c_i max(0, 1-y_i(v^T z_RBF(x_i)+b))/sum_i c_i"
    )
    print(
        "Logistic stacking ensemble:\n"
        "  J_stack = weighted BCE(y, sigmoid(a^T s+b)) "
        "+ (lambda/2)||a||^2"
    )
    print("=" * 84)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def labels_to_signed(y: np.ndarray, positive_class: Any) -> np.ndarray:
    return np.where(y == positive_class, 1.0, -1.0).astype(np.float32)


def make_class_weight_map(
    y: np.ndarray,
    classes: np.ndarray,
    positive_multiplier: float,
) -> dict[Any, float]:
    balanced = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    result = {
        class_label: float(weight)
        for class_label, weight in zip(classes, balanced)
    }
    result[classes[1]] *= positive_multiplier
    return result


def make_sample_weights(
    y: np.ndarray,
    classes: np.ndarray,
    positive_multiplier: float,
    balanced: bool = True,
) -> np.ndarray:
    if not balanced:
        weights = np.ones(len(y), dtype=np.float32)
        weights[y == classes[1]] *= positive_multiplier
        return weights

    mapping = make_class_weight_map(y, classes, positive_multiplier)
    return np.asarray([mapping[value] for value in y], dtype=np.float32)


def weighted_hinge_loss_from_scores(
    y_signed: np.ndarray,
    scores: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[float, float]:
    hinge = np.maximum(0.0, 1.0 - y_signed * scores)
    denominator = max(float(sample_weights.sum()), 1e-12)
    loss = float(np.dot(sample_weights, hinge) / denominator)
    violation_rate = float(np.mean(hinge > 0.0))
    return loss, violation_rate


def primal_hinge_components(
    X: np.ndarray,
    y_signed: np.ndarray,
    sample_weights: np.ndarray,
    weights: np.ndarray,
    bias: float,
    regularization: float,
) -> tuple[float, float, float, float]:
    scores = X @ weights + bias
    hinge_loss, violation_rate = weighted_hinge_loss_from_scores(
        y_signed,
        scores,
        sample_weights,
    )
    regularization_loss = 0.5 * regularization * float(weights @ weights)
    total_loss = hinge_loss + regularization_loss
    return total_loss, hinge_loss, regularization_loss, violation_rate


def weighted_binary_cross_entropy(
    y_binary: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    individual = -(
        y_binary * np.log(probabilities)
        + (1.0 - y_binary) * np.log(1.0 - probabilities)
    )
    denominator = max(float(sample_weights.sum()), 1e-12)
    return float(np.dot(sample_weights, individual) / denominator)


def stratified_subsample_indices(
    y: np.ndarray,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    y = np.asarray(y)
    if len(y) <= max_samples:
        return np.arange(len(y))

    indices = np.arange(len(y))
    sampled, _ = train_test_split(
        indices,
        train_size=max_samples,
        stratify=y,
        random_state=random_state,
    )
    return np.asarray(sampled)


def safe_standardize_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum <= 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


# ============================================================================
# THRESHOLD SELECTION
# ============================================================================

def select_multi_objective_threshold(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    positive_class: Any,
    mode: str,
    target_recall: float,
    min_precision: float,
    beta: float,
    recall_penalty: float,
    precision_penalty: float,
) -> dict[str, Any]:
    y_binary = (np.asarray(y_true) == positive_class).astype(int)

    precision_values, recall_values, thresholds = precision_recall_curve(
        y_binary,
        decision_scores,
    )

    precision_values = precision_values[:-1]
    recall_values = recall_values[:-1]

    beta_squared = beta * beta
    f_beta_values = (
        (1.0 + beta_squared)
        * precision_values
        * recall_values
        / (
            beta_squared * precision_values
            + recall_values
            + 1e-12
        )
    )

    if len(thresholds) == 0:
        raise ValueError("Cannot select threshold from an empty curve.")

    if mode == "max_f1":
        best_index = int(np.argmax(f_beta_values))
        selection_mode = "maximum_f_beta"
    elif mode == "constraints":
        valid_mask = (
            (recall_values >= target_recall)
            & (precision_values >= min_precision)
        )

        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            objective = (
                f_beta_values[valid_indices]
                + 1e-6 * precision_values[valid_indices]
            )
            best_index = int(valid_indices[np.argmax(objective)])
            selection_mode = "constraints_satisfied"
        else:
            recall_shortfall = np.maximum(
                0.0,
                target_recall - recall_values,
            )
            precision_shortfall = np.maximum(
                0.0,
                min_precision - precision_values,
            )
            objective = (
                f_beta_values
                - recall_penalty * recall_shortfall
                - precision_penalty * precision_shortfall
            )
            best_index = int(np.argmax(objective))
            selection_mode = "best_penalized_compromise"
    else:
        raise ValueError(
            "THRESHOLD_MODE must be either 'constraints' or 'max_f1'."
        )

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision_values[best_index]),
        "recall": float(recall_values[best_index]),
        "f_beta": float(f_beta_values[best_index]),
        "mode": selection_mode,
        "curve": pd.DataFrame(
            {
                "threshold": thresholds,
                "precision": precision_values,
                "recall": recall_values,
                "f_beta": f_beta_values,
            }
        ),
    }


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def encode_binary_target(y_raw: pd.Series, target_name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y_raw):
        values = sorted(pd.Series(y_raw.dropna().unique()).tolist())
        if len(values) != 2:
            raise ValueError(
                f"Target '{target_name}' must contain exactly two values; "
                f"found {len(values)}: {values[:10]}"
            )
        if set(values) == {0, 1}:
            return y_raw.astype(int)
        mapping = {values[0]: 0, values[1]: 1}
        print(f"  Numeric target mapping: {mapping}")
        return y_raw.map(mapping).astype(int)

    normalized = y_raw.astype(str).str.strip().str.lower()
    positive_keywords = (
        "dead",
        "died",
        "deceased",
        "yes",
        "malignant",
        "positive",
        "event",
    )
    negative_keywords = (
        "alive",
        "living",
        "no",
        "benign",
        "negative",
        "censored",
    )

    encoded = pd.Series(index=y_raw.index, dtype=float)
    for idx, value in normalized.items():
        if any(keyword in value for keyword in positive_keywords):
            encoded.loc[idx] = 1
        elif any(keyword in value for keyword in negative_keywords):
            encoded.loc[idx] = 0

    unresolved = encoded.isna()
    if unresolved.any():
        unresolved_values = sorted(normalized[unresolved].unique())
        if len(unresolved_values) != 2:
            raise ValueError(
                "Cannot automatically encode target values: "
                f"{unresolved_values[:20]}"
            )
        mapping = {
            unresolved_values[0]: 0,
            unresolved_values[1]: 1,
        }
        print(f"  Fallback target mapping: {mapping}")
        encoded.loc[unresolved] = normalized[unresolved].map(mapping)

    return encoded.astype(int)


def load_seer_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    print("\n" + "=" * 84)
    print("STEP 1 — LOADING DATA")
    print("=" * 84)

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")
    print(f"  First columns: {list(df.columns[:10])}")

    target_col = next(
        (column for column in SEER_TARGET_CANDIDATES if column in df.columns),
        None,
    )
    if target_col is None:
        raise ValueError(
            "Target column not found. Add its name to SEER_TARGET_CANDIDATES."
        )

    df = df.loc[df[target_col].notna()].copy()
    y = encode_binary_target(df[target_col], target_col)

    drop_columns = [target_col, *BASE_DROP_COLUMNS]
    if DROP_SURVIVAL_DERIVED_FEATURES:
        drop_columns.extend(SURVIVAL_DERIVED_COLUMNS)
    if DROP_DIAGNOSIS_TIME_FEATURES:
        drop_columns.extend(DIAGNOSIS_TIME_COLUMNS)

    drop_columns = [column for column in drop_columns if column in df.columns]
    X = df.drop(columns=drop_columns)

    print(f"  Target column: {target_col}")
    print(f"  Dropped columns: {drop_columns}")
    print(f"  Feature shape: {X.shape}")
    print(f"  Target distribution: {y.value_counts().sort_index().to_dict()}")
    return X, y


class SEERPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, missing_threshold: float = 0.50):
        self.missing_threshold = missing_threshold

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        keep_columns = X.columns[
            X.isna().mean() < self.missing_threshold
        ].tolist()
        keep_columns = [
            column
            for column in keep_columns
            if X[column].nunique(dropna=True) > 1
        ]

        self.feature_names_ = keep_columns
        X = X[self.feature_names_].copy()

        self.categorical_columns_ = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        self.numeric_columns_ = [
            column
            for column in self.feature_names_
            if column not in self.categorical_columns_
        ]

        self.numeric_fill_values_ = {}
        for column in self.numeric_columns_:
            numeric = pd.to_numeric(X[column], errors="coerce")
            median = numeric.median()
            self.numeric_fill_values_[column] = (
                0.0 if pd.isna(median) else float(median)
            )

        self.categorical_fill_values_ = {}
        for column in self.categorical_columns_:
            mode = X[column].mode(dropna=True)
            self.categorical_fill_values_[column] = (
                str(mode.iloc[0]) if not mode.empty else "__MISSING__"
            )

        if self.categorical_columns_:
            categorical = X[self.categorical_columns_].copy()
            for column in self.categorical_columns_:
                categorical[column] = (
                    categorical[column]
                    .fillna(self.categorical_fill_values_[column])
                    .astype(str)
                )
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.encoder_.fit(categorical)
        else:
            self.encoder_ = None

        print("\n" + "=" * 84)
        print("STEP 3 — TRAIN-ONLY PREPROCESSING")
        print("=" * 84)
        print(f"  Kept features        : {len(self.feature_names_)}")
        print(f"  Numeric features     : {len(self.numeric_columns_)}")
        print(f"  Categorical features : {len(self.categorical_columns_)}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "feature_names_")
        X = X.copy()

        for column in self.feature_names_:
            if column not in X.columns:
                X[column] = np.nan
        X = X[self.feature_names_]

        transformed = pd.DataFrame(
            index=X.index,
            columns=self.feature_names_,
            dtype=float,
        )

        for column in self.numeric_columns_:
            transformed[column] = (
                pd.to_numeric(X[column], errors="coerce")
                .fillna(self.numeric_fill_values_[column])
                .astype(float)
            )

        if self.categorical_columns_:
            categorical = X[self.categorical_columns_].copy()
            for column in self.categorical_columns_:
                categorical[column] = (
                    categorical[column]
                    .fillna(self.categorical_fill_values_[column])
                    .astype(str)
                )
            transformed.loc[:, self.categorical_columns_] = (
                self.encoder_.transform(categorical)
            )

        return transformed[self.feature_names_].to_numpy(dtype=np.float32)


def run_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    k: int,
    max_samples: int,
) -> tuple[np.ndarray, list[str], SelectKBest]:
    print("\n" + "=" * 84)
    print("STEP 4 — MUTUAL INFORMATION FEATURE SELECTION")
    print("=" * 84)

    indices = stratified_subsample_indices(
        y_train,
        max_samples=max_samples,
        random_state=RANDOM_STATE,
    )
    selector = SelectKBest(
        score_func=partial(
            mutual_info_classif,
            random_state=RANDOM_STATE,
        ),
        k=min(k, X_train.shape[1]),
    )

    start = time.time()
    selector.fit(X_train[indices], np.asarray(y_train)[indices])
    elapsed = time.time() - start

    mask = selector.get_support()
    selected_names = [
        name for name, selected in zip(feature_names, mask) if selected
    ]
    scores = np.nan_to_num(selector.scores_, nan=-np.inf)
    ranking = sorted(
        zip(feature_names, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    print(f"  MI sample size    : {len(indices):,}")
    print(f"  Original features : {len(feature_names)}")
    print(f"  Selected features : {len(selected_names)}")
    print(f"  Selection time    : {elapsed:.2f}s")
    print("  Top features:")
    for name, score in ranking[: min(12, len(ranking))]:
        print(f"    {name:<48} {score:.6f}")

    return mask, selected_names, selector


def split_training_roles(
    X: pd.DataFrame,
    y: pd.Series,
    meta_fraction: float,
    threshold_fraction: float,
    random_state: int,
):
    if meta_fraction + threshold_fraction >= 1.0:
        raise ValueError("meta_fraction + threshold_fraction must be < 1.")

    X_remaining, X_threshold, y_remaining, y_threshold = train_test_split(
        X,
        y,
        test_size=threshold_fraction,
        stratify=y,
        random_state=random_state,
    )

    relative_meta_fraction = meta_fraction / (1.0 - threshold_fraction)
    X_base, X_meta, y_base, y_meta = train_test_split(
        X_remaining,
        y_remaining,
        test_size=relative_meta_fraction,
        stratify=y_remaining,
        random_state=random_state + 1,
    )

    return X_base, X_meta, X_threshold, y_base, y_meta, y_threshold



# ============================================================================
# LIGHTWEIGHT HYPERPARAMETER TUNING
# ============================================================================

def maximum_f1_summary(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    positive_class: Any,
) -> dict[str, float]:
    result = select_multi_objective_threshold(
        y_true=y_true,
        decision_scores=decision_scores,
        positive_class=positive_class,
        mode="max_f1",
        target_recall=0.0,
        min_precision=0.0,
        beta=1.0,
        recall_penalty=0.0,
        precision_penalty=0.0,
    )
    return {
        "threshold": float(result["threshold"]),
        "precision": float(result["precision"]),
        "recall": float(result["recall"]),
        "f1": float(result["f_beta"]),
        "pr_auc": float(
            average_precision_score(
                (np.asarray(y_true) == positive_class).astype(int),
                decision_scores,
            )
        ),
    }


def _candidate_is_better(
    candidate: dict[str, Any],
    current_best: dict[str, Any] | None,
) -> bool:
    if current_best is None:
        return True

    # Primary criterion: PR-AUC, because it evaluates ranking quality
    # independently of one chosen threshold. F1 breaks near ties.
    candidate_key = (
        round(float(candidate["pr_auc"]), 8),
        round(float(candidate["f1"]), 8),
    )
    best_key = (
        round(float(current_best["pr_auc"]), 8),
        round(float(current_best["f1"]), 8),
    )
    return candidate_key > best_key


def run_lightweight_tuning(
    X_base: np.ndarray,
    y_base: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Tune the three base learners on a stratified subset of the base-training set.

    The untouched meta, threshold-validation, and test roles are not used here.
    """
    global LINEAR_SVM_LAMBDA
    global LINEAR_SVM_INITIAL_LR
    global LINEAR_SVM_EPOCHS
    global RBF_GAMMA
    global RBF_COMPONENTS
    global RBF_SVM_LAMBDA
    global RBF_SVM_EPOCHS
    global CVM_C
    global CVM_GAMMA

    print("\n" + "=" * 84)
    print("STEP 5B — LIGHTWEIGHT PR-AUC / MAX-F1 TUNING")
    print("=" * 84)

    sampled_indices = stratified_subsample_indices(
        y_base,
        max_samples=min(TUNING_MAX_SAMPLES, len(y_base)),
        random_state=TUNING_RANDOM_STATE,
    )
    X_sample = X_base[sampled_indices]
    y_sample = np.asarray(y_base)[sampled_indices]

    X_tune_train, X_tune_valid, y_tune_train, y_tune_valid = train_test_split(
        X_sample,
        y_sample,
        test_size=TUNING_VALIDATION_SIZE,
        stratify=y_sample,
        random_state=TUNING_RANDOM_STATE,
    )

    positive_class = np.unique(y_tune_train)[1]
    rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Linear SGD-SVM
    # ------------------------------------------------------------------
    best_linear: dict[str, Any] | None = None
    for index, candidate in enumerate(LINEAR_TUNING_CANDIDATES, start=1):
        start_time = time.time()
        model = LinearMiniBatchSGDSVM(
            regularization=float(candidate["regularization"]),
            initial_lr=float(candidate["initial_lr"]),
            epochs=int(candidate["epochs"]),
            batch_size=LINEAR_SVM_BATCH_SIZE,
            positive_multiplier=1.0,
            random_state=RANDOM_STATE + index,
            print_every_update=False,
            print_every_n_updates=10_000,
        ).fit(X_tune_train, y_tune_train)

        scores = model.decision_function(X_tune_valid)
        metrics = maximum_f1_summary(y_tune_valid, scores, positive_class)

        row = {
            "model": "Linear-SGD-SVM",
            "candidate": index,
            **candidate,
            **metrics,
            "fit_seconds": time.time() - start_time,
        }
        rows.append(row)

        if _candidate_is_better(row, best_linear):
            best_linear = row

        print(
            f"  Linear candidate {index}: "
            f"lambda={candidate['regularization']} "
            f"epochs={candidate['epochs']} "
            f"PR-AUC={metrics['pr_auc']:.4f} "
            f"max-F1={metrics['f1']:.4f}"
        )

    assert best_linear is not None
    LINEAR_SVM_LAMBDA = float(best_linear["regularization"])
    LINEAR_SVM_INITIAL_LR = float(best_linear["initial_lr"])
    LINEAR_SVM_EPOCHS = int(best_linear["epochs"])

    # ------------------------------------------------------------------
    # RBF-Sampler SGD-SVM
    # ------------------------------------------------------------------
    best_rbf: dict[str, Any] | None = None
    for index, candidate in enumerate(RBF_TUNING_CANDIDATES, start=1):
        start_time = time.time()
        model = RBFSamplerMiniBatchSGDSVM(
            gamma=float(candidate["gamma"]),
            n_components=int(candidate["n_components"]),
            regularization=float(candidate["regularization"]),
            initial_lr=RBF_SVM_INITIAL_LR,
            epochs=int(candidate["epochs"]),
            batch_size=RBF_SVM_BATCH_SIZE,
            transform_chunk_size=RBF_TRANSFORM_CHUNK_SIZE,
            precompute_train=True,
            positive_multiplier=1.0,
            random_state=RANDOM_STATE + 100 + index,
            print_every_update=False,
            print_every_n_updates=10_000,
        ).fit(X_tune_train, y_tune_train)

        scores = model.decision_function(X_tune_valid)
        metrics = maximum_f1_summary(y_tune_valid, scores, positive_class)

        row = {
            "model": "RBF-Sampler-SGD-SVM",
            "candidate": index,
            **candidate,
            **metrics,
            "fit_seconds": time.time() - start_time,
        }
        rows.append(row)

        if _candidate_is_better(row, best_rbf):
            best_rbf = row

        print(
            f"  RBF candidate {index}: "
            f"gamma={candidate['gamma']} "
            f"components={candidate['n_components']} "
            f"PR-AUC={metrics['pr_auc']:.4f} "
            f"max-F1={metrics['f1']:.4f}"
        )

    assert best_rbf is not None
    RBF_GAMMA = float(best_rbf["gamma"])
    RBF_COMPONENTS = int(best_rbf["n_components"])
    RBF_SVM_LAMBDA = float(best_rbf["regularization"])
    RBF_SVM_EPOCHS = int(best_rbf["epochs"])

    # ------------------------------------------------------------------
    # CVM-style RBF-SVM
    # ------------------------------------------------------------------
    pilot = SharedBoundaryPilot(
        C=PILOT_C,
        positive_multiplier=1.0,
        random_state=RANDOM_STATE,
    ).fit(X_tune_train, y_tune_train)

    best_cvm: dict[str, Any] | None = None
    for index, candidate in enumerate(CVM_TUNING_CANDIDATES, start=1):
        start_time = time.time()
        model = CVMStyleRBFSVM(
            core_stages=CVM_TUNING_CORE_STAGES,
            C=float(candidate["C"]),
            gamma=float(candidate["gamma"]),
            cache_size_mb=CVM_CACHE_SIZE_MB,
            positive_multiplier=1.0,
            balanced_core=CVM_BALANCED_CORE,
            random_state=RANDOM_STATE + 200 + index,
            print_every_update=False,
        ).fit_with_pilot(
            X_tune_train,
            y_tune_train,
            pilot.distances_,
            X_tune_valid,
            y_tune_valid,
        )

        scores = model.decision_function(X_tune_valid)
        metrics = maximum_f1_summary(y_tune_valid, scores, positive_class)

        row = {
            "model": "CVM-RBF",
            "candidate": index,
            **candidate,
            **metrics,
            "fit_seconds": time.time() - start_time,
        }
        rows.append(row)

        if _candidate_is_better(row, best_cvm):
            best_cvm = row

        print(
            f"  CVM candidate {index}: "
            f"C={candidate['C']} gamma={candidate['gamma']} "
            f"PR-AUC={metrics['pr_auc']:.4f} "
            f"max-F1={metrics['f1']:.4f}"
        )

    assert best_cvm is not None
    CVM_C = float(best_cvm["C"])
    CVM_GAMMA = float(best_cvm["gamma"])

    tuning_results = pd.DataFrame(rows)
    tuning_results = tuning_results.sort_values(
        by=["model", "pr_auc", "f1"],
        ascending=[True, False, False],
    )
    tuning_path = output_dir / "lightweight_tuning_results.csv"
    tuning_results.to_csv(tuning_path, index=False)

    print("\n  Selected hyperparameters:")
    print(
        f"    Linear SGD: lambda={LINEAR_SVM_LAMBDA}, "
        f"lr={LINEAR_SVM_INITIAL_LR}, epochs={LINEAR_SVM_EPOCHS}"
    )
    print(
        f"    RBF SGD   : gamma={RBF_GAMMA}, "
        f"components={RBF_COMPONENTS}, "
        f"lambda={RBF_SVM_LAMBDA}, epochs={RBF_SVM_EPOCHS}"
    )
    print(f"    CVM-RBF   : C={CVM_C}, gamma={CVM_GAMMA}")
    print(f"    Tuning CSV: {tuning_path}")

    return {
        "results": tuning_results,
        "csv_path": tuning_path,
        "best_linear": best_linear,
        "best_rbf": best_rbf,
        "best_cvm": best_cvm,
    }


# ============================================================================
# BASE MODEL 1: CVM-STYLE RBF-SVC
# ============================================================================

class SharedBoundaryPilot:
    def __init__(
        self,
        C: float,
        positive_multiplier: float,
        random_state: int,
    ):
        self.C = C
        self.positive_multiplier = positive_multiplier
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        class_weight = make_class_weight_map(
            y,
            self.classes_,
            self.positive_multiplier,
        )
        self.model_ = LinearSVC(
            C=self.C,
            class_weight=class_weight,
            dual="auto",
            max_iter=3_000,
            tol=1e-3,
            random_state=self.random_state,
        )
        start = time.time()
        self.model_.fit(X, y)
        self.distances_ = np.abs(self.model_.decision_function(X))
        print(f"  Shared pilot fit time: {time.time() - start:.2f}s")
        return self


class CVMStyleRBFSVM(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        core_stages: tuple[int, ...],
        C: float,
        gamma: str | float,
        cache_size_mb: int,
        positive_multiplier: float,
        balanced_core: bool,
        random_state: int,
        print_every_update: bool,
    ):
        self.core_stages = core_stages
        self.C = C
        self.gamma = gamma
        self.cache_size_mb = cache_size_mb
        self.positive_multiplier = positive_multiplier
        self.balanced_core = balanced_core
        self.random_state = random_state
        self.print_every_update = print_every_update

    def _select_core_indices(
        self,
        y: np.ndarray,
        distances: np.ndarray,
        n_core: int,
    ) -> np.ndarray:
        classes = np.unique(y)
        if not self.balanced_core:
            indices = np.argsort(distances)[:n_core]
            return np.asarray(indices, dtype=int)

        selected_parts = []
        remaining = n_core
        for class_index, class_label in enumerate(classes):
            class_indices = np.where(y == class_label)[0]
            if class_index == len(classes) - 1:
                class_count = min(remaining, len(class_indices))
            else:
                class_count = min(n_core // len(classes), len(class_indices))
                remaining -= class_count
            order = np.argsort(distances[class_indices])[:class_count]
            selected_parts.append(class_indices[order])

        selected = np.concatenate(selected_parts)
        if len(selected) < n_core:
            not_selected_mask = np.ones(len(y), dtype=bool)
            not_selected_mask[selected] = False
            candidates = np.where(not_selected_mask)[0]
            extra_order = np.argsort(distances[candidates])[: n_core - len(selected)]
            selected = np.concatenate([selected, candidates[extra_order]])

        return np.asarray(selected[:n_core], dtype=int)

    def fit_with_pilot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        monitor_X: np.ndarray,
        monitor_y: np.ndarray,
    ):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        class_weight = make_class_weight_map(
            y,
            self.classes_,
            self.positive_multiplier,
        )
        monitor_signed = labels_to_signed(monitor_y, self.classes_[1])
        monitor_weights = make_sample_weights(
            monitor_y,
            self.classes_,
            self.positive_multiplier,
            balanced=True,
        )

        valid_stages = sorted(
            {
                min(int(stage), len(X))
                for stage in self.core_stages
                if int(stage) >= 2
            }
        )
        if not valid_stages:
            valid_stages = [min(len(X), 500)]

        self.loss_history_ = []
        print("\n  Training CVM-style RBF-SVM by progressive core size...")

        for update, n_core in enumerate(valid_stages, start=1):
            core_indices = self._select_core_indices(y, distances, n_core)
            model = SVC(
                C=self.C,
                kernel="rbf",
                gamma=self.gamma,
                class_weight=class_weight,
                probability=False,
                cache_size=self.cache_size_mb,
                random_state=self.random_state,
            )

            start = time.time()
            model.fit(X[core_indices], y[core_indices])
            elapsed = time.time() - start

            monitor_scores = model.decision_function(monitor_X)
            hinge_loss, violation_rate = weighted_hinge_loss_from_scores(
                monitor_signed,
                monitor_scores,
                monitor_weights,
            )

            row = {
                "update": update,
                "n_core": len(core_indices),
                "monitor_hinge_loss": hinge_loss,
                "violation_rate": violation_rate,
                "support_vectors": int(model.n_support_.sum()),
                "fit_seconds": elapsed,
            }
            self.loss_history_.append(row)

            if self.print_every_update:
                print(
                    f"    [CVM-RBF] update={update:02d} "
                    f"core={len(core_indices):4d} "
                    f"hinge={hinge_loss:.6f} "
                    f"viol={violation_rate:.4f} "
                    f"SV={row['support_vectors']:4d} "
                    f"time={elapsed:.2f}s"
                )

            self.model_ = model
            self.core_indices_ = core_indices

        self.loss_history_df_ = pd.DataFrame(self.loss_history_)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        pilot = SharedBoundaryPilot(
            C=PILOT_C,
            positive_multiplier=self.positive_multiplier,
            random_state=self.random_state,
        ).fit(X, y)
        monitor_indices = stratified_subsample_indices(
            y,
            max_samples=min(LOSS_MONITOR_SAMPLES, len(y)),
            random_state=self.random_state,
        )
        return self.fit_with_pilot(
            X,
            y,
            pilot.distances_,
            X[monitor_indices],
            y[monitor_indices],
        )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_function(X)
        return np.where(score >= 0.0, self.classes_[1], self.classes_[0])


# ============================================================================
# BASE MODEL 2: LINEAR MINI-BATCH PRIMAL SGD-SVM
# ============================================================================

class LinearMiniBatchSGDSVM(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        regularization: float,
        initial_lr: float,
        epochs: int,
        batch_size: int,
        positive_multiplier: float,
        random_state: int,
        print_every_update: bool,
        print_every_n_updates: int,
    ):
        self.regularization = regularization
        self.initial_lr = initial_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.positive_multiplier = positive_multiplier
        self.random_state = random_state
        self.print_every_update = print_every_update
        self.print_every_n_updates = print_every_n_updates

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_signed = labels_to_signed(y, self.classes_[1])
        sample_weights = make_sample_weights(
            y,
            self.classes_,
            self.positive_multiplier,
            balanced=True,
        )

        rng = np.random.default_rng(self.random_state)
        self.coef_ = np.zeros(X.shape[1], dtype=np.float32)
        self.intercept_ = 0.0
        self.loss_history_ = []
        update = 0

        print("\n  Training linear Mini-Batch SGD-SVM...")
        for epoch in range(1, self.epochs + 1):
            indices = rng.permutation(len(X))
            for batch_number, start in enumerate(
                range(0, len(X), self.batch_size),
                start=1,
            ):
                update += 1
                batch_indices = indices[start : start + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_signed[batch_indices]
                weights_batch = sample_weights[batch_indices]

                scores = X_batch @ self.coef_ + self.intercept_
                violating = y_batch * scores < 1.0
                denominator = max(float(weights_batch.sum()), 1e-12)

                gradient_w = self.regularization * self.coef_
                gradient_b = 0.0
                if np.any(violating):
                    effective = weights_batch[violating] * y_batch[violating]
                    gradient_w -= (
                        X_batch[violating].T @ effective
                    ) / denominator
                    gradient_b -= float(effective.sum() / denominator)

                learning_rate = self.initial_lr / math.sqrt(update)
                self.coef_ -= learning_rate * gradient_w.astype(np.float32)
                self.intercept_ -= learning_rate * gradient_b

                total, hinge, reg, violation_rate = primal_hinge_components(
                    X_batch,
                    y_batch,
                    weights_batch,
                    self.coef_,
                    self.intercept_,
                    self.regularization,
                )

                row = {
                    "update": update,
                    "epoch": epoch,
                    "batch": batch_number,
                    "learning_rate": learning_rate,
                    "hinge_loss": hinge,
                    "regularization_loss": reg,
                    "total_loss": total,
                    "violation_rate": violation_rate,
                }
                self.loss_history_.append(row)

                if (
                    self.print_every_update
                    and update % self.print_every_n_updates == 0
                ):
                    print(
                        f"    [Linear-SGD] update={update:04d} "
                        f"epoch={epoch:02d} batch={batch_number:03d} "
                        f"lr={learning_rate:.6f} "
                        f"hinge={hinge:.6f} reg={reg:.6f} "
                        f"total={total:.6f} viol={violation_rate:.4f}"
                    )

        self.loss_history_df_ = pd.DataFrame(self.loss_history_)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=np.float32) @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_function(X)
        return np.where(score >= 0.0, self.classes_[1], self.classes_[0])


# ============================================================================
# BASE MODEL 3: RBFSAMPLER + MINI-BATCH PRIMAL SGD-SVM
# ============================================================================

class RBFSamplerMiniBatchSGDSVM(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        gamma: float,
        n_components: int,
        regularization: float,
        initial_lr: float,
        epochs: int,
        batch_size: int,
        transform_chunk_size: int,
        precompute_train: bool,
        positive_multiplier: float,
        random_state: int,
        print_every_update: bool,
        print_every_n_updates: int,
    ):
        self.gamma = gamma
        self.n_components = n_components
        self.regularization = regularization
        self.initial_lr = initial_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.transform_chunk_size = transform_chunk_size
        self.precompute_train = precompute_train
        self.positive_multiplier = positive_multiplier
        self.random_state = random_state
        self.print_every_update = print_every_update
        self.print_every_n_updates = print_every_n_updates

    def _transform_in_chunks(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        transformed = np.empty(
            (len(X), self.n_components),
            dtype=np.float32,
        )
        for start in range(0, len(X), self.transform_chunk_size):
            end = min(start + self.transform_chunk_size, len(X))
            transformed[start:end] = self.mapper_.transform(
                X[start:end]
            ).astype(np.float32)
        return transformed

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_signed = labels_to_signed(y, self.classes_[1])
        sample_weights = make_sample_weights(
            y,
            self.classes_,
            self.positive_multiplier,
            balanced=True,
        )

        self.mapper_ = RBFSampler(
            gamma=self.gamma,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.mapper_.fit(X[:1])

        print("\n  Preparing Random Fourier Features...")
        transform_start = time.time()
        if self.precompute_train:
            transformed_train = self._transform_in_chunks(X)
            print(
                f"    Precomputed shape={transformed_train.shape}, "
                f"memory≈{transformed_train.nbytes / 1024**2:.1f} MB, "
                f"time={time.time() - transform_start:.2f}s"
            )
        else:
            transformed_train = None
            print("    RBF features will be transformed per mini-batch.")

        rng = np.random.default_rng(self.random_state)
        self.coef_ = np.zeros(self.n_components, dtype=np.float32)
        self.intercept_ = 0.0
        self.loss_history_ = []
        update = 0

        print("  Training RBF-Sampler Mini-Batch SGD-SVM...")
        for epoch in range(1, self.epochs + 1):
            indices = rng.permutation(len(X))
            for batch_number, start in enumerate(
                range(0, len(X), self.batch_size),
                start=1,
            ):
                update += 1
                batch_indices = indices[start : start + self.batch_size]
                if transformed_train is None:
                    Z_batch = self.mapper_.transform(
                        X[batch_indices]
                    ).astype(np.float32)
                else:
                    Z_batch = transformed_train[batch_indices]

                y_batch = y_signed[batch_indices]
                weights_batch = sample_weights[batch_indices]
                scores = Z_batch @ self.coef_ + self.intercept_
                violating = y_batch * scores < 1.0
                denominator = max(float(weights_batch.sum()), 1e-12)

                gradient_w = self.regularization * self.coef_
                gradient_b = 0.0
                if np.any(violating):
                    effective = weights_batch[violating] * y_batch[violating]
                    gradient_w -= (
                        Z_batch[violating].T @ effective
                    ) / denominator
                    gradient_b -= float(effective.sum() / denominator)

                learning_rate = self.initial_lr / math.sqrt(update)
                self.coef_ -= learning_rate * gradient_w.astype(np.float32)
                self.intercept_ -= learning_rate * gradient_b

                total, hinge, reg, violation_rate = primal_hinge_components(
                    Z_batch,
                    y_batch,
                    weights_batch,
                    self.coef_,
                    self.intercept_,
                    self.regularization,
                )

                row = {
                    "update": update,
                    "epoch": epoch,
                    "batch": batch_number,
                    "learning_rate": learning_rate,
                    "hinge_loss": hinge,
                    "regularization_loss": reg,
                    "total_loss": total,
                    "violation_rate": violation_rate,
                }
                self.loss_history_.append(row)

                if (
                    self.print_every_update
                    and update % self.print_every_n_updates == 0
                ):
                    print(
                        f"    [RBF-SGD] update={update:04d} "
                        f"epoch={epoch:02d} batch={batch_number:03d} "
                        f"lr={learning_rate:.6f} "
                        f"hinge={hinge:.6f} reg={reg:.6f} "
                        f"total={total:.6f} viol={violation_rate:.4f}"
                    )

        self.loss_history_df_ = pd.DataFrame(self.loss_history_)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float32)
        scores = np.empty(len(X), dtype=np.float32)
        for start in range(0, len(X), self.transform_chunk_size):
            end = min(start + self.transform_chunk_size, len(X))
            Z = self.mapper_.transform(X[start:end]).astype(np.float32)
            scores[start:end] = Z @ self.coef_ + self.intercept_
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_function(X)
        return np.where(score >= 0.0, self.classes_[1], self.classes_[0])


# ============================================================================
# LOGISTIC SGD STACKER
# ============================================================================

class LogisticSGDStacker(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        regularization: float,
        initial_lr: float,
        epochs: int,
        batch_size: int,
        use_balanced_weights: bool,
        positive_multiplier: float,
        random_state: int,
        print_every_update: bool,
        print_every_n_updates: int,
    ):
        self.regularization = regularization
        self.initial_lr = initial_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_balanced_weights = use_balanced_weights
        self.positive_multiplier = positive_multiplier
        self.random_state = random_state
        self.print_every_update = print_every_update
        self.print_every_n_updates = print_every_n_updates

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_binary = (y == self.classes_[1]).astype(np.float32)
        sample_weights = make_sample_weights(
            y,
            self.classes_,
            self.positive_multiplier,
            balanced=self.use_balanced_weights,
        )

        rng = np.random.default_rng(self.random_state)
        self.coef_ = np.zeros(X.shape[1], dtype=np.float32)
        self.intercept_ = 0.0
        self.loss_history_ = []
        update = 0

        print("\n  Training logistic SGD stacking model...")
        for epoch in range(1, self.epochs + 1):
            indices = rng.permutation(len(X))
            for batch_number, start in enumerate(
                range(0, len(X), self.batch_size),
                start=1,
            ):
                update += 1
                batch_indices = indices[start : start + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_binary[batch_indices]
                weights_batch = sample_weights[batch_indices]

                logits = X_batch @ self.coef_ + self.intercept_
                probabilities = sigmoid(logits)
                denominator = max(float(weights_batch.sum()), 1e-12)
                error = (probabilities - y_batch) * weights_batch

                gradient_w = (
                    X_batch.T @ error
                ) / denominator + self.regularization * self.coef_
                gradient_b = float(error.sum() / denominator)

                learning_rate = self.initial_lr / math.sqrt(update)
                self.coef_ -= learning_rate * gradient_w.astype(np.float32)
                self.intercept_ -= learning_rate * gradient_b

                updated_probabilities = sigmoid(
                    X_batch @ self.coef_ + self.intercept_
                )
                bce = weighted_binary_cross_entropy(
                    y_batch,
                    updated_probabilities,
                    weights_batch,
                )
                reg = 0.5 * self.regularization * float(
                    self.coef_ @ self.coef_
                )
                total = bce + reg

                row = {
                    "update": update,
                    "epoch": epoch,
                    "batch": batch_number,
                    "learning_rate": learning_rate,
                    "binary_cross_entropy": bce,
                    "regularization_loss": reg,
                    "total_loss": total,
                }
                self.loss_history_.append(row)

                if (
                    self.print_every_update
                    and update % self.print_every_n_updates == 0
                ):
                    print(
                        f"    [Stacker] update={update:04d} "
                        f"epoch={epoch:03d} batch={batch_number:03d} "
                        f"lr={learning_rate:.6f} "
                        f"BCE={bce:.6f} reg={reg:.6f} total={total:.6f}"
                    )

        self.loss_history_df_ = pd.DataFrame(self.loss_history_)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=np.float32) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_function(X)
        return np.where(score >= 0.0, self.classes_[1], self.classes_[0])


# ============================================================================
# STACKED ENSEMBLE
# ============================================================================

class CVMSGDRBFStackingEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    @staticmethod
    def _score_matrix(models: list[Any], X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.asarray(model.decision_function(X)).ravel() for model in models]
        ).astype(np.float32)

    def fit(
        self,
        X_base: np.ndarray,
        y_base: np.ndarray,
        X_meta: np.ndarray,
        y_meta: np.ndarray,
        X_threshold: np.ndarray,
        y_threshold: np.ndarray,
    ):
        print("\n" + "=" * 84)
        print("STEP 6 — TRAINING THREE BASE MODELS")
        print("=" * 84)

        self.classes_ = np.unique(y_base)

        pilot = SharedBoundaryPilot(
            C=PILOT_C,
            positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER,
            random_state=self.random_state,
        ).fit(X_base, y_base)

        monitor_indices = stratified_subsample_indices(
            y_base,
            max_samples=min(LOSS_MONITOR_SAMPLES, len(y_base)),
            random_state=self.random_state,
        )

        self.cvm_rbf_ = CVMStyleRBFSVM(
            core_stages=CVM_CORE_STAGES,
            C=CVM_C,
            gamma=CVM_GAMMA,
            cache_size_mb=CVM_CACHE_SIZE_MB,
            positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER,
            balanced_core=CVM_BALANCED_CORE,
            random_state=self.random_state,
            print_every_update=PRINT_EVERY_UPDATE,
        ).fit_with_pilot(
            X_base,
            y_base,
            pilot.distances_,
            X_base[monitor_indices],
            y_base[monitor_indices],
        )

        self.linear_sgd_ = LinearMiniBatchSGDSVM(
            regularization=LINEAR_SVM_LAMBDA,
            initial_lr=LINEAR_SVM_INITIAL_LR,
            epochs=LINEAR_SVM_EPOCHS,
            batch_size=LINEAR_SVM_BATCH_SIZE,
            positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER,
            random_state=self.random_state,
            print_every_update=PRINT_EVERY_UPDATE,
            print_every_n_updates=PRINT_EVERY_N_UPDATES,
        ).fit(X_base, y_base)

        self.rbf_sgd_ = RBFSamplerMiniBatchSGDSVM(
            gamma=RBF_GAMMA,
            n_components=RBF_COMPONENTS,
            regularization=RBF_SVM_LAMBDA,
            initial_lr=RBF_SVM_INITIAL_LR,
            epochs=RBF_SVM_EPOCHS,
            batch_size=RBF_SVM_BATCH_SIZE,
            transform_chunk_size=RBF_TRANSFORM_CHUNK_SIZE,
            precompute_train=RBF_PRECOMPUTE_TRAIN,
            positive_multiplier=POSITIVE_WEIGHT_MULTIPLIER,
            random_state=self.random_state,
            print_every_update=PRINT_EVERY_UPDATE,
            print_every_n_updates=PRINT_EVERY_N_UPDATES,
        ).fit(X_base, y_base)

        self.base_models_ = [
            self.cvm_rbf_,
            self.linear_sgd_,
            self.rbf_sgd_,
        ]
        self.model_names_ = [
            "CVM-RBF",
            "Linear-SGD-SVM",
            "RBF-Sampler-SGD-SVM",
        ]

        print("\n" + "=" * 84)
        print("STEP 7 — LOGISTIC STACKING")
        print("=" * 84)

        meta_scores = self._score_matrix(self.base_models_, X_meta)
        self.score_scaler_ = StandardScaler()
        meta_scores_scaled = self.score_scaler_.fit_transform(meta_scores).astype(
            np.float32
        )

        self.stacker_ = LogisticSGDStacker(
            regularization=STACK_LAMBDA,
            initial_lr=STACK_INITIAL_LR,
            epochs=STACK_EPOCHS,
            batch_size=STACK_BATCH_SIZE,
            use_balanced_weights=STACK_USE_BALANCED_WEIGHTS,
            positive_multiplier=1.0,
            random_state=self.random_state,
            print_every_update=PRINT_EVERY_UPDATE,
            print_every_n_updates=PRINT_EVERY_N_UPDATES,
        ).fit(meta_scores_scaled, y_meta)

        print("\n  Stacking coefficients:")
        for name, coefficient in zip(
            self.model_names_,
            self.stacker_.coef_,
        ):
            print(f"    {name:<24}: {float(coefficient): .6f}")
        print(f"    Intercept               : {self.stacker_.intercept_: .6f}")

        print("\n" + "=" * 84)
        print("STEP 8 — MAXIMUM-F1 THRESHOLD SELECTION")
        print("=" * 84)

        threshold_scores_base = self._score_matrix(
            self.base_models_,
            X_threshold,
        )
        threshold_scores_scaled = self.score_scaler_.transform(
            threshold_scores_base
        ).astype(np.float32)
        threshold_decision = self.stacker_.decision_function(
            threshold_scores_scaled
        )

        threshold_result = select_multi_objective_threshold(
            y_true=y_threshold,
            decision_scores=threshold_decision,
            positive_class=self.classes_[1],
            mode=THRESHOLD_MODE,
            target_recall=TARGET_RECALL,
            min_precision=MIN_PRECISION,
            beta=THRESHOLD_BETA,
            recall_penalty=RECALL_SHORTFALL_PENALTY,
            precision_penalty=PRECISION_SHORTFALL_PENALTY,
        )

        self.threshold_ = threshold_result["threshold"]
        self.threshold_precision_ = threshold_result["precision"]
        self.threshold_recall_ = threshold_result["recall"]
        self.threshold_f_beta_ = threshold_result["f_beta"]
        self.threshold_mode_ = threshold_result["mode"]
        self.threshold_curve_ = threshold_result["curve"]

        print(f"  Selection mode : {self.threshold_mode_}")
        print(f"  Threshold      : {self.threshold_:.6f}")
        print(f"  Precision      : {self.threshold_precision_:.4f}")
        print(f"  Recall         : {self.threshold_recall_:.4f}")
        print(f"  F-beta         : {self.threshold_f_beta_:.4f}")

        # Per-base-model threshold-validation diagnostics.
        base_metric_rows = []
        for model_name, model in zip(self.model_names_, self.base_models_):
            model_scores = np.asarray(
                model.decision_function(X_threshold)
            ).ravel()
            summary = maximum_f1_summary(
                y_threshold,
                model_scores,
                self.classes_[1],
            )
            base_metric_rows.append(
                {
                    "model": model_name,
                    **summary,
                }
            )
        self.base_validation_metrics_ = pd.DataFrame(base_metric_rows)

        print("\n  Base-model threshold-validation diagnostics:")
        print(
            self.base_validation_metrics_[
                ["model", "precision", "recall", "f1", "pr_auc"]
            ].to_string(index=False)
        )

        # Diversity diagnostics on threshold-validation data.
        self.score_correlation_ = pd.DataFrame(
            threshold_scores_base,
            columns=self.model_names_,
        ).corr()

        predictions = np.column_stack(
            [model.predict(X_threshold) for model in self.base_models_]
        )
        disagreement = np.zeros((len(self.model_names_), len(self.model_names_)))
        for i in range(len(self.model_names_)):
            for j in range(len(self.model_names_)):
                disagreement[i, j] = np.mean(predictions[:, i] != predictions[:, j])
        self.disagreement_matrix_ = pd.DataFrame(
            disagreement,
            index=self.model_names_,
            columns=self.model_names_,
        )

        return self

    def base_score_matrix(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "base_models_")
        return self._score_matrix(self.base_models_, X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        base_scores = self.base_score_matrix(X)
        scaled = self.score_scaler_.transform(base_scores).astype(np.float32)
        return self.stacker_.decision_function(scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_function(X)
        return np.where(
            score >= self.threshold_,
            self.classes_[1],
            self.classes_[0],
        )


# ============================================================================
# EVALUATION, PLOTS, AND EXPORTS
# ============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    label: str,
) -> dict[str, Any]:
    print("\n" + "─" * 84)
    print(f"{label} — TEST EVALUATION")
    print("─" * 84)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Alive/0", "Dead/1"],
            zero_division=0,
        )
    )

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    print(f"  ROC-AUC : {results['roc_auc']:.4f}")
    print(f"  PR-AUC  : {results['pr_auc']:.4f}")
    print(f"  Confusion matrix:\n{results['confusion_matrix']}")
    return results


def save_line_plot(
    dataframe: pd.DataFrame,
    x_column: str,
    y_columns: list[str],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    for column in y_columns:
        if column in dataframe.columns:
            plt.plot(dataframe[x_column], dataframe[column], label=column)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    if len(y_columns) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def export_outputs(
    ensemble: CVMSGDRBFStackingEnsemble,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    histories = {
        "cvm_rbf": ensemble.cvm_rbf_.loss_history_df_,
        "linear_sgd": ensemble.linear_sgd_.loss_history_df_,
        "rbf_sgd": ensemble.rbf_sgd_.loss_history_df_,
        "stacker": ensemble.stacker_.loss_history_df_,
    }

    for name, dataframe in histories.items():
        csv_path = output_dir / f"{name}_loss_history.csv"
        dataframe.to_csv(csv_path, index=False)
        paths[f"{name}_csv"] = csv_path

    save_line_plot(
        histories["cvm_rbf"],
        x_column="n_core",
        y_columns=["monitor_hinge_loss"],
        title="CVM-RBF monitoring hinge loss by core-set size",
        x_label="Number of core vectors",
        y_label="Weighted hinge loss",
        output_path=output_dir / "cvm_rbf_loss_curve.png",
    )
    paths["cvm_rbf_plot"] = output_dir / "cvm_rbf_loss_curve.png"

    save_line_plot(
        histories["linear_sgd"],
        x_column="update",
        y_columns=["total_loss", "hinge_loss"],
        title="Linear Mini-Batch SGD-SVM loss",
        x_label="Update",
        y_label="Loss",
        output_path=output_dir / "linear_sgd_loss_curve.png",
    )
    paths["linear_sgd_plot"] = output_dir / "linear_sgd_loss_curve.png"

    save_line_plot(
        histories["rbf_sgd"],
        x_column="update",
        y_columns=["total_loss", "hinge_loss"],
        title="RBF-Sampler Mini-Batch SGD-SVM loss",
        x_label="Update",
        y_label="Loss",
        output_path=output_dir / "rbf_sgd_loss_curve.png",
    )
    paths["rbf_sgd_plot"] = output_dir / "rbf_sgd_loss_curve.png"

    save_line_plot(
        histories["stacker"],
        x_column="update",
        y_columns=["total_loss", "binary_cross_entropy"],
        title="Logistic stacking ensemble loss",
        x_label="Update",
        y_label="Loss",
        output_path=output_dir / "stacker_loss_curve.png",
    )
    paths["stacker_plot"] = output_dir / "stacker_loss_curve.png"

    threshold_csv = output_dir / "threshold_metrics.csv"
    ensemble.threshold_curve_.to_csv(threshold_csv, index=False)
    paths["threshold_csv"] = threshold_csv

    save_line_plot(
        ensemble.threshold_curve_,
        x_column="threshold",
        y_columns=["precision", "recall", "f_beta"],
        title="Precision, recall, and F-beta by stacking threshold",
        x_label="Decision threshold",
        y_label="Metric",
        output_path=output_dir / "threshold_metrics_curve.png",
    )
    paths["threshold_plot"] = output_dir / "threshold_metrics_curve.png"

    correlation_path = output_dir / "base_score_correlation.csv"
    ensemble.score_correlation_.to_csv(correlation_path)
    paths["correlation_csv"] = correlation_path

    disagreement_path = output_dir / "base_prediction_disagreement.csv"
    ensemble.disagreement_matrix_.to_csv(disagreement_path)
    paths["disagreement_csv"] = disagreement_path

    # Combined normalized loss view. Different losses have different scales,
    # so normalize each curve independently before visual comparison.
    normalized_curves = []
    for model_name, dataframe, x_column, loss_column in [
        ("CVM-RBF", histories["cvm_rbf"], "n_core", "monitor_hinge_loss"),
        ("Linear-SGD", histories["linear_sgd"], "update", "total_loss"),
        ("RBF-SGD", histories["rbf_sgd"], "update", "total_loss"),
        ("Stacker", histories["stacker"], "update", "total_loss"),
    ]:
        values = dataframe[loss_column].to_numpy(dtype=float)
        progress = np.linspace(0.0, 1.0, len(values))
        normalized = safe_standardize_series(values)
        normalized_curves.append((model_name, progress, normalized))

    plt.figure(figsize=(10, 6))
    for model_name, progress, normalized in normalized_curves:
        plt.plot(progress, normalized, label=model_name)
    plt.title("Normalized loss progression across all models")
    plt.xlabel("Normalized training progress")
    plt.ylabel("Independently normalized loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    combined_path = output_dir / "all_normalized_loss_curves.png"
    plt.savefig(combined_path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    paths["combined_loss_plot"] = combined_path

    return paths


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(
    data_path: str = DATA_PATH,
    output_dir: Path | str = OUTPUT_DIR,
    quick_test: bool = False,
) -> dict[str, Any]:
    global CVM_CORE_STAGES
    global LINEAR_SVM_EPOCHS
    global RBF_COMPONENTS
    global RBF_SVM_EPOCHS
    global STACK_EPOCHS
    global TOP_K_FEATURES
    global MI_MAX_SAMPLES
    global PRINT_EVERY_UPDATE
    global LINEAR_SVM_LAMBDA
    global LINEAR_SVM_INITIAL_LR
    global LINEAR_SVM_EPOCHS
    global RBF_GAMMA
    global RBF_COMPONENTS
    global RBF_SVM_LAMBDA
    global RBF_SVM_EPOCHS
    global CVM_C
    global CVM_GAMMA

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if quick_test:
        CVM_CORE_STAGES = (100, 200, 400)
        LINEAR_SVM_EPOCHS = 3
        RBF_COMPONENTS = 128
        RBF_SVM_EPOCHS = 3
        STACK_EPOCHS = 15
        TOP_K_FEATURES = min(TOP_K_FEATURES, 10)
        MI_MAX_SAMPLES = 5_000
        PRINT_EVERY_UPDATE = False

    total_start = time.time()
    print_loss_definitions()

    X_all, y_all = load_seer_data(data_path)

    print("\n" + "=" * 84)
    print("STEP 2 — CLEAN DATA SPLITTING")
    print("=" * 84)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all,
        y_all,
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )

    (
        X_base,
        X_meta,
        X_threshold,
        y_base,
        y_meta,
        y_threshold,
    ) = split_training_roles(
        X_train_full,
        y_train_full,
        meta_fraction=META_FRACTION,
        threshold_fraction=THRESHOLD_FRACTION,
        random_state=RANDOM_STATE,
    )

    print(f"  Base train      : {X_base.shape}")
    print(f"  Meta train      : {X_meta.shape}")
    print(f"  Threshold valid : {X_threshold.shape}")
    print(f"  Untouched test  : {X_test.shape}")

    preprocessor = SEERPreprocessor(missing_threshold=0.50)
    X_base_encoded = preprocessor.fit_transform(X_base)
    X_meta_encoded = preprocessor.transform(X_meta)
    X_threshold_encoded = preprocessor.transform(X_threshold)
    X_test_encoded = preprocessor.transform(X_test)

    feature_mask, selected_names, selector = run_feature_selection(
        X_base_encoded,
        y_base.to_numpy(),
        preprocessor.feature_names_,
        k=TOP_K_FEATURES,
        max_samples=MI_MAX_SAMPLES,
    )

    X_base_selected = X_base_encoded[:, feature_mask]
    X_meta_selected = X_meta_encoded[:, feature_mask]
    X_threshold_selected = X_threshold_encoded[:, feature_mask]
    X_test_selected = X_test_encoded[:, feature_mask]

    print("\n" + "=" * 84)
    print("STEP 5 — STANDARD SCALING")
    print("=" * 84)

    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base_selected).astype(np.float32)
    X_meta_scaled = scaler.transform(X_meta_selected).astype(np.float32)
    X_threshold_scaled = scaler.transform(X_threshold_selected).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_selected).astype(np.float32)

    print(f"  Base scaled shape      : {X_base_scaled.shape}")
    print(f"  Meta scaled shape      : {X_meta_scaled.shape}")
    print(f"  Threshold scaled shape : {X_threshold_scaled.shape}")
    print(f"  Test scaled shape      : {X_test_scaled.shape}")

    tuning_artifacts = None
    if RUN_LIGHTWEIGHT_TUNING and not quick_test:
        tuning_artifacts = run_lightweight_tuning(
            X_base_scaled,
            y_base.to_numpy(),
            output_dir,
        )

    ensemble = CVMSGDRBFStackingEnsemble(random_state=RANDOM_STATE)
    ensemble.fit(
        X_base_scaled,
        y_base.to_numpy(),
        X_meta_scaled,
        y_meta.to_numpy(),
        X_threshold_scaled,
        y_threshold.to_numpy(),
    )

    print("\n" + "=" * 84)
    print("STEP 9 — UNTOUCHED TEST PREDICTION")
    print("=" * 84)

    y_pred = ensemble.predict(X_test_scaled)
    y_score = ensemble.decision_function(X_test_scaled)

    results = evaluate_model(
        y_test.to_numpy(),
        y_pred,
        y_score,
        label=(
            "CVM-RBF + Linear Mini-Batch SGD-SVM + "
            "RBF-Sampler SGD-SVM Stacking"
        ),
    )

    output_paths = export_outputs(ensemble, output_dir)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 84)
    print("SUMMARY")
    print("=" * 84)
    print(f"  Selected features: {len(selected_names)}")
    print(f"  Feature names    : {selected_names}")
    print("  Selected hyperparameters:")
    print(
        f"    CVM-RBF      : C={CVM_C}, gamma={CVM_GAMMA}, "
        f"core stages={CVM_CORE_STAGES}"
    )
    print(
        f"    Linear SGD   : lambda={LINEAR_SVM_LAMBDA}, "
        f"epochs={LINEAR_SVM_EPOCHS}, lr={LINEAR_SVM_INITIAL_LR}"
    )
    print(
        f"    RBF SGD      : gamma={RBF_GAMMA}, "
        f"components={RBF_COMPONENTS}, "
        f"lambda={RBF_SVM_LAMBDA}, epochs={RBF_SVM_EPOCHS}"
    )
    print("  Stacking coefficients:")
    for name, coefficient in zip(
        ensemble.model_names_,
        ensemble.stacker_.coef_,
    ):
        print(f"    {name:<24}: {float(coefficient): .6f}")
    print(f"  Threshold        : {ensemble.threshold_:.6f}")
    print(f"  Threshold mode   : {ensemble.threshold_mode_}")
    print(
        "  Validation P/R/F : "
        f"{ensemble.threshold_precision_:.4f} / "
        f"{ensemble.threshold_recall_:.4f} / "
        f"{ensemble.threshold_f_beta_:.4f}"
    )
    print(f"  Accuracy         : {results['accuracy']:.4f}")
    print(f"  Precision        : {results['precision']:.4f}")
    print(f"  Recall           : {results['recall']:.4f}")
    print(f"  F1-score         : {results['f1']:.4f}")
    print(f"  ROC-AUC          : {results['roc_auc']:.4f}")
    print(f"  PR-AUC           : {results['pr_auc']:.4f}")
    print(f"  Total runtime    : {total_elapsed / 60:.2f} minutes")
    print(f"  Outputs          : {output_dir}")
    print("=" * 84)

    print("\n  Base score correlation:")
    print(ensemble.score_correlation_.round(4))
    print("\n  Base prediction disagreement:")
    print(ensemble.disagreement_matrix_.round(4))

    artifacts = {
        "preprocessor": preprocessor,
        "feature_selector": selector,
        "selected_feature_names": selected_names,
        "scaler": scaler,
        "ensemble": ensemble,
        "results": results,
        "output_paths": output_paths,
        "tuning_artifacts": tuning_artifacts,
    }

    if SAVE_MODEL_ARTIFACTS:
        try:
            import joblib
            model_path = output_dir / "seer_stacking_artifacts.joblib"
            joblib.dump(artifacts, model_path)
            print(f"  Saved model artifacts: {model_path}")
        except Exception as exc:
            print(f"  [WARN] Could not save model artifacts: {exc}")

    return artifacts


if __name__ == "__main__":
    artifacts = main(DATA_PATH, OUTPUT_DIR)
