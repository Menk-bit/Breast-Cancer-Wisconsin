from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# CONFIG
# =========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "full": REPO_ROOT / "data" / "model_ready_scaled.csv",
}

# Chọn dataset để chạy:
# "all_valid", "chart_selected", hoặc "clinical_selected"
DATASET_NAME = "full"

TARGET_COL = "survive_after_5"

LEAKAGE_COLS = [
    "survive_after_5",
    "survival_months_int",
    "survival_months_unknown_flag",
]

RANDOM_STATE = 42

TEST_SIZE = 0.2
VAL_SIZE = 0.25
# 0.25 của phần train_val tương đương:
# train = 60%, validation = 20%, test = 20%

MAX_EPOCHS = 1000
PATIENCE = 50
MIN_DELTA = 1e-5

LEARNING_RATES = [0.01, 0.005, 0.001]
LAMBDA_L2_VALUES = [0.0, 0.001, 0.01]
USE_CLASS_WEIGHT_VALUES = [False, True]
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

OUTPUT_DIR = Path("logistic_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# DATA
# =========================================================

def load_dataset(path):
    df = pd.read_csv(path)

    y = df[TARGET_COL].astype(int).values

    X = df.drop(columns=LEAKAGE_COLS, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_names = X.columns.tolist()
    X = X.values.astype(float)

    return X, y, feature_names


def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================================================
# LOGISTIC REGRESSION FROM SCRATCH
# =========================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_class_weights(y):
    n = len(y)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    weight_pos = n / (2 * n_pos)
    weight_neg = n / (2 * n_neg)

    return weight_neg, weight_pos


def binary_cross_entropy(y_true, y_prob, sample_weights=None):
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)

    losses = -(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )

    if sample_weights is not None:
        return np.average(losses, weights=sample_weights)

    return np.mean(losses)


def compute_loss(X, y, w, b, lambda_l2, sample_weights=None):
    y_prob = sigmoid(X @ w + b)

    data_loss = binary_cross_entropy(
        y_true=y,
        y_prob=y_prob,
        sample_weights=sample_weights
    )

    l2_loss = 0.5 * lambda_l2 * np.sum(w ** 2)

    return data_loss + l2_loss


def train_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate,
    lambda_l2,
    use_class_weight,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE
):
    n_samples, n_features = X_train.shape

    rng = np.random.default_rng(RANDOM_STATE)
    w = rng.normal(loc=0.0, scale=0.01, size=n_features)
    b = 0.0

    if use_class_weight:
        weight_neg, weight_pos = compute_class_weights(y_train)
        train_weights = np.where(y_train == 1, weight_pos, weight_neg)
        val_weights = np.where(y_val == 1, weight_pos, weight_neg)
    else:
        train_weights = np.ones_like(y_train, dtype=float)
        val_weights = np.ones_like(y_val, dtype=float)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = np.inf
    best_w = w.copy()
    best_b = b
    no_improve_count = 0

    for epoch in range(1, max_epochs + 1):
        # Forward
        logits = X_train @ w + b
        y_prob = sigmoid(logits)

        # Weighted error
        error = y_prob - y_train
        weighted_error = train_weights * error

        # Gradient
        grad_w = (X_train.T @ weighted_error) / n_samples
        grad_b = np.sum(weighted_error) / n_samples

        # L2 regularization gradient
        grad_w += lambda_l2 * w

        # Update
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        # Loss
        train_loss = compute_loss(
            X_train,
            y_train,
            w,
            b,
            lambda_l2,
            sample_weights=train_weights
        )

        val_loss = compute_loss(
            X_val,
            y_val,
            w,
            b,
            lambda_l2,
            sample_weights=val_weights
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_w = w.copy()
            best_b = b
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    return best_w, best_b, history


# =========================================================
# EVALUATION
# =========================================================

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def predict_label(X, w, b, threshold):
    prob = predict_proba(X, w, b)
    return (prob >= threshold).astype(int)


def evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_dead": precision_score(y_true, y_pred, zero_division=0),
        "recall_dead": recall_score(y_true, y_pred, zero_division=0),
        "f1_dead": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    return result


def tune_threshold(y_val, y_val_prob):
    rows = []

    for threshold in THRESHOLDS:
        metrics = evaluate_predictions(
            y_true=y_val,
            y_prob=y_val_prob,
            threshold=threshold
        )
        rows.append(metrics)

    threshold_df = pd.DataFrame(rows)

    # Ưu tiên F1 cho class Dead.
    # Nếu muốn ưu tiên giảm FN hơn nữa, có thể đổi sang recall_dead.
    threshold_df = threshold_df.sort_values(
        ["f1_dead", "recall_dead", "roc_auc"],
        ascending=False
    )

    best_threshold = float(threshold_df.iloc[0]["threshold"])

    return best_threshold, threshold_df


# =========================================================
# PLOTS
# =========================================================

def plot_loss_curve(history, output_path):
    history_df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")

    plt.title("Logistic Regression training curve")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Alive", "Pred Dead"],
        yticklabels=["True Alive", "True Dead"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_threshold_tuning(threshold_df, output_path):
    plot_df = threshold_df.sort_values("threshold")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["threshold"], plot_df["precision_dead"], marker="o", label="Precision Dead")
    plt.plot(plot_df["threshold"], plot_df["recall_dead"], marker="o", label="Recall Dead")
    plt.plot(plot_df["threshold"], plot_df["f1_dead"], marker="o", label="F1 Dead")

    plt.title("Threshold tuning on validation set")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_feature_coefficients(feature_names, w, output_path):
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": w,
        "abs_coefficient": np.abs(w)
    })

    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(output_path, index=False, encoding="utf-8-sig")


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

def run_hyperparameter_search(X_train, X_val, y_train, y_val):
    tuning_rows = []
    model_store = {}

    config_id = 0

    for lr in LEARNING_RATES:
        for lambda_l2 in LAMBDA_L2_VALUES:
            for use_class_weight in USE_CLASS_WEIGHT_VALUES:
                config_id += 1

                w, b, history = train_logistic_regression(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    learning_rate=lr,
                    lambda_l2=lambda_l2,
                    use_class_weight=use_class_weight
                )

                y_val_prob = predict_proba(X_val, w, b)

                best_threshold, threshold_df = tune_threshold(
                    y_val=y_val,
                    y_val_prob=y_val_prob
                )

                val_metrics = evaluate_predictions(
                    y_true=y_val,
                    y_prob=y_val_prob,
                    threshold=best_threshold
                )

                row = {
                    "config_id": config_id,
                    "learning_rate": lr,
                    "lambda_l2": lambda_l2,
                    "use_class_weight": use_class_weight,
                    "epochs_trained": len(history["epoch"]),
                    "best_threshold": best_threshold,
                    "final_train_loss": history["train_loss"][-1],
                    "final_val_loss": history["val_loss"][-1],
                    **val_metrics,
                }

                tuning_rows.append(row)

                model_store[config_id] = {
                    "w": w,
                    "b": b,
                    "history": history,
                    "threshold_df": threshold_df,
                }

                print(
                    f"Config {config_id:02d} | "
                    f"lr={lr}, l2={lambda_l2}, class_weight={use_class_weight} | "
                    f"val_f1={val_metrics['f1_dead']:.4f}, "
                    f"val_recall={val_metrics['recall_dead']:.4f}, "
                    f"val_auc={val_metrics['roc_auc']:.4f}, "
                    f"epochs={len(history['epoch'])}"
                )

    tuning_df = pd.DataFrame(tuning_rows)

    tuning_df = tuning_df.sort_values(
        ["f1_dead", "recall_dead", "roc_auc"],
        ascending=False
    )

    best_config_id = int(tuning_df.iloc[0]["config_id"])

    return tuning_df, model_store, best_config_id


# =========================================================
# MAIN
# =========================================================

def main():
    input_path = DATA_FILES[DATASET_NAME]

    print("=" * 100)
    print("LOGISTIC REGRESSION FROM SCRATCH")
    print("=" * 100)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Input file: {input_path}")

    X, y, feature_names = load_dataset(input_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("\nData split:")
    print(f"- Train:      {X_train.shape}")
    print(f"- Validation: {X_val.shape}")
    print(f"- Test:       {X_test.shape}")

    print("\nClass distribution:")
    print(f"- Train Dead ratio:      {y_train.mean():.4f}")
    print(f"- Validation Dead ratio: {y_val.mean():.4f}")
    print(f"- Test Dead ratio:       {y_test.mean():.4f}")

    print("\nRunning hyperparameter search...")
    tuning_df, model_store, best_config_id = run_hyperparameter_search(
        X_train,
        X_val,
        y_train,
        y_val
    )

    tuning_path = OUTPUT_DIR / f"{DATASET_NAME}_tuning_results.csv"
    tuning_df.to_csv(tuning_path, index=False, encoding="utf-8-sig")

    best_row = tuning_df.iloc[0]
    best_model = model_store[best_config_id]

    best_w = best_model["w"]
    best_b = best_model["b"]
    best_history = best_model["history"]
    best_threshold = float(best_row["best_threshold"])

    print("\n" + "=" * 100)
    print("BEST CONFIG ON VALIDATION")
    print("=" * 100)
    print(best_row.to_string())

    # Final evaluation on test set
    y_test_prob = predict_proba(X_test, best_w, best_b)
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    test_metrics = evaluate_predictions(
        y_true=y_test,
        y_prob=y_test_prob,
        threshold=best_threshold
    )

    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_path = OUTPUT_DIR / f"{DATASET_NAME}_test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("FINAL TEST METRICS")
    print("=" * 100)
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=["Alive", "Dead"],
        zero_division=0
    ))

    # Save plots
    plot_loss_curve(
        best_history,
        OUTPUT_DIR / f"{DATASET_NAME}_loss_curve_best_model.png"
    )

    plot_confusion_matrix(
        y_test,
        y_test_pred,
        OUTPUT_DIR / f"{DATASET_NAME}_confusion_matrix.png"
    )

    plot_threshold_tuning(
        best_model["threshold_df"],
        OUTPUT_DIR / f"{DATASET_NAME}_threshold_tuning.png"
    )

    save_feature_coefficients(
        feature_names,
        best_w,
        OUTPUT_DIR / f"{DATASET_NAME}_feature_coefficients.csv"
    )

    print("\n" + "=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)
    print(f"- {tuning_path}")
    print(f"- {test_metrics_path}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_loss_curve_best_model.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_confusion_matrix.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_threshold_tuning.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_feature_coefficients.csv'}")


if __name__ == "__main__":
    main()
