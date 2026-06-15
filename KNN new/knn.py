from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
BASE_DIR = REPO_ROOT / "data"

DATA_FILES = {
    "full": BASE_DIR / "model_ready_scaled.csv",
}

# Chọn 1 bộ data để chạy:
# "all_valid", "chart_selected", hoặc "clinical_selected"
DATASET_NAME = "full"

TARGET_COL = "survive_after_5"

LEAKAGE_COLS = [
    "survive_after_5",
    "survival_months_int",
    "survival_months_unknown_flag",
]

OUTPUT_DIR = Path("knn_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

TEST_SIZE = 0.2
VAL_SIZE = 0.25
# train = 60%, validation = 20%, test = 20%

K_VALUES = [3, 5, 7, 9, 11, 15]

WEIGHT_OPTIONS = [
    "uniform",
    "distance",
]

DISTANCE_OPTIONS = [
    {"metric_name": "manhattan", "metric": "minkowski", "p": 1},
    {"metric_name": "euclidean", "metric": "minkowski", "p": 2},
]

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# KNN trên 200k dòng có thể rất chậm.
# Để None nếu muốn chạy full data.
MAX_TUNING_TRAIN_ROWS = 50000
MAX_TUNING_VAL_ROWS = None
MAX_FINAL_TRAIN_ROWS = None


# =========================================================
# DATA
# =========================================================

def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    df = pd.read_csv(path)

    y = df[TARGET_COL].astype(int)

    X = df.drop(columns=LEAKAGE_COLS, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            "File vẫn còn cột không phải numeric:\n"
            + "\n".join(non_numeric_cols)
        )

    feature_names = X.columns.tolist()

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

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val


def stratified_sample(X, y, max_rows):
    if max_rows is None or len(X) <= max_rows:
        return X, y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_sample, y_sample


def to_numpy_float32(X):
    return X.values.astype(np.float32)


# =========================================================
# METRICS
# =========================================================

def evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_dead": precision_score(y_true, y_pred, zero_division=0),
        "recall_dead": recall_score(y_true, y_pred, zero_division=0),
        "f1_dead": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


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

    threshold_df = threshold_df.sort_values(
        ["f1_dead", "recall_dead", "roc_auc"],
        ascending=False
    )

    best_threshold = float(threshold_df.iloc[0]["threshold"])

    return best_threshold, threshold_df


# =========================================================
# KNN MODEL
# =========================================================

def build_knn(k, weights, metric, p):
    return KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        metric=metric,
        p=p,
        algorithm="auto",
        n_jobs=-1
    )


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

def run_hyperparameter_search(X_train, y_train, X_val, y_val):
    X_train_tune, y_train_tune = stratified_sample(
        X_train,
        y_train,
        MAX_TUNING_TRAIN_ROWS
    )

    X_val_tune, y_val_tune = stratified_sample(
        X_val,
        y_val,
        MAX_TUNING_VAL_ROWS
    )

    X_train_np = to_numpy_float32(X_train_tune)
    X_val_np = to_numpy_float32(X_val_tune)

    y_train_np = y_train_tune.values.astype(int)
    y_val_np = y_val_tune.values.astype(int)

    tuning_rows = []
    threshold_store = {}

    config_id = 0

    for k in K_VALUES:
        for weights in WEIGHT_OPTIONS:
            for dist in DISTANCE_OPTIONS:
                config_id += 1

                metric_name = dist["metric_name"]
                metric = dist["metric"]
                p = dist["p"]

                model = build_knn(
                    k=k,
                    weights=weights,
                    metric=metric,
                    p=p
                )

                model.fit(X_train_np, y_train_np)

                y_train_prob = model.predict_proba(X_train_np)[:, 1]
                y_val_prob = model.predict_proba(X_val_np)[:, 1]

                best_threshold, threshold_df = tune_threshold(
                    y_val=y_val_np,
                    y_val_prob=y_val_prob
                )

                train_metrics = evaluate_predictions(
                    y_true=y_train_np,
                    y_prob=y_train_prob,
                    threshold=best_threshold
                )

                val_metrics = evaluate_predictions(
                    y_true=y_val_np,
                    y_prob=y_val_prob,
                    threshold=best_threshold
                )

                train_val_f1_gap = abs(
                    train_metrics["f1_dead"] - val_metrics["f1_dead"]
                )

                row = {
                    "config_id": config_id,
                    "k": k,
                    "weights": weights,
                    "metric_name": metric_name,
                    "metric": metric,
                    "p": p,
                    "best_threshold": best_threshold,

                    "train_accuracy": train_metrics["accuracy"],
                    "train_precision_dead": train_metrics["precision_dead"],
                    "train_recall_dead": train_metrics["recall_dead"],
                    "train_f1_dead": train_metrics["f1_dead"],
                    "train_roc_auc": train_metrics["roc_auc"],

                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision_dead": val_metrics["precision_dead"],
                    "val_recall_dead": val_metrics["recall_dead"],
                    "val_f1_dead": val_metrics["f1_dead"],
                    "val_roc_auc": val_metrics["roc_auc"],

                    "train_val_f1_gap": train_val_f1_gap,
                }

                tuning_rows.append(row)
                threshold_store[config_id] = threshold_df

                print(
                    f"Config {config_id:02d} | "
                    f"k={k}, weights={weights}, metric={metric_name} | "
                    f"val_f1={val_metrics['f1_dead']:.4f}, "
                    f"val_recall={val_metrics['recall_dead']:.4f}, "
                    f"val_auc={val_metrics['roc_auc']:.4f}, "
                    f"gap={train_val_f1_gap:.4f}"
                )

    tuning_df = pd.DataFrame(tuning_rows)

    tuning_df = tuning_df.sort_values(
        ["val_f1_dead", "val_recall_dead", "val_roc_auc", "train_val_f1_gap"],
        ascending=[False, False, False, True]
    )

    best_config_id = int(tuning_df.iloc[0]["config_id"])

    return tuning_df, threshold_store, best_config_id


# =========================================================
# FINAL TEST
# =========================================================

def final_evaluate_best_model(best_row, X_train_val, y_train_val, X_test, y_test):
    X_final_train, y_final_train = stratified_sample(
        X_train_val,
        y_train_val,
        MAX_FINAL_TRAIN_ROWS
    )

    X_final_np = to_numpy_float32(X_final_train)
    X_test_np = to_numpy_float32(X_test)

    y_final_np = y_final_train.values.astype(int)
    y_test_np = y_test.values.astype(int)

    model = build_knn(
        k=int(best_row["k"]),
        weights=best_row["weights"],
        metric=best_row["metric"],
        p=int(best_row["p"])
    )

    model.fit(X_final_np, y_final_np)

    y_test_prob = model.predict_proba(X_test_np)[:, 1]
    threshold = float(best_row["best_threshold"])
    y_test_pred = (y_test_prob >= threshold).astype(int)

    test_metrics = evaluate_predictions(
        y_true=y_test_np,
        y_prob=y_test_prob,
        threshold=threshold
    )

    return model, test_metrics, y_test_pred, y_test_prob, y_test_np


# =========================================================
# PLOTS
# =========================================================

def plot_validation_curve(tuning_df, output_path):
    plot_df = tuning_df.copy()
    plot_df["setting"] = plot_df["weights"] + " + " + plot_df["metric_name"]

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=plot_df,
        x="k",
        y="val_f1_dead",
        hue="setting",
        marker="o"
    )

    plt.title("KNN validation F1-score by k")
    plt.xlabel("Number of neighbors k")
    plt.ylabel("Validation F1-score for Dead")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_train_val_gap(tuning_df, output_path):
    plot_df = tuning_df.copy()
    plot_df["setting"] = plot_df["weights"] + " + " + plot_df["metric_name"]

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=plot_df,
        x="k",
        y="train_val_f1_gap",
        hue="setting",
        marker="o"
    )

    plt.title("KNN train-validation F1 gap by k")
    plt.xlabel("Number of neighbors k")
    plt.ylabel("Absolute train-validation F1 gap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_threshold_tuning(threshold_df, output_path):
    plot_df = threshold_df.sort_values("threshold")

    plt.figure(figsize=(8, 5))
    plt.plot(
        plot_df["threshold"],
        plot_df["precision_dead"],
        marker="o",
        label="Precision Dead"
    )
    plt.plot(
        plot_df["threshold"],
        plot_df["recall_dead"],
        marker="o",
        label="Recall Dead"
    )
    plt.plot(
        plot_df["threshold"],
        plot_df["f1_dead"],
        marker="o",
        label="F1 Dead"
    )

    plt.title("KNN threshold tuning on validation set")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
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

    plt.title("KNN Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():
    input_path = DATA_FILES[DATASET_NAME]

    print("=" * 100)
    print("KNN PIPELINE")
    print("=" * 100)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Input file: {input_path}")

    X, y, feature_names = load_dataset(input_path)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_val,
        y_train_val
    ) = split_data(X, y)

    print("\nData split:")
    print(f"- Train:      {X_train.shape}")
    print(f"- Validation: {X_val.shape}")
    print(f"- Test:       {X_test.shape}")

    print("\nClass distribution:")
    print(f"- Train Dead ratio:      {y_train.mean():.4f}")
    print(f"- Validation Dead ratio: {y_val.mean():.4f}")
    print(f"- Test Dead ratio:       {y_test.mean():.4f}")

    print("\nRunning hyperparameter search...")
    tuning_df, threshold_store, best_config_id = run_hyperparameter_search(
        X_train,
        y_train,
        X_val,
        y_val
    )

    tuning_path = OUTPUT_DIR / f"{DATASET_NAME}_tuning_results.csv"
    tuning_df.to_csv(tuning_path, index=False, encoding="utf-8-sig")

    best_row = tuning_df.iloc[0]
    best_threshold_df = threshold_store[best_config_id]

    threshold_path = OUTPUT_DIR / f"{DATASET_NAME}_threshold_tuning_best.csv"
    best_threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("BEST CONFIG ON VALIDATION")
    print("=" * 100)
    print(best_row.to_string())

    (
        model,
        test_metrics,
        y_test_pred,
        y_test_prob,
        y_test_np
    ) = final_evaluate_best_model(
        best_row=best_row,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        y_test=y_test
    )

    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_path = OUTPUT_DIR / f"{DATASET_NAME}_test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_path, index=False, encoding="utf-8-sig")

    report_path = OUTPUT_DIR / f"{DATASET_NAME}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(
            y_test_np,
            y_test_pred,
            target_names=["Alive", "Dead"],
            zero_division=0
        ))

    print("\n" + "=" * 100)
    print("FINAL TEST METRICS")
    print("=" * 100)
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nClassification report:")
    print(classification_report(
        y_test_np,
        y_test_pred,
        target_names=["Alive", "Dead"],
        zero_division=0
    ))

    plot_validation_curve(
        tuning_df,
        OUTPUT_DIR / f"{DATASET_NAME}_validation_curve_f1.png"
    )

    plot_train_val_gap(
        tuning_df,
        OUTPUT_DIR / f"{DATASET_NAME}_train_val_gap.png"
    )

    plot_threshold_tuning(
        best_threshold_df,
        OUTPUT_DIR / f"{DATASET_NAME}_threshold_tuning.png"
    )

    plot_confusion_matrix(
        y_test_np,
        y_test_pred,
        OUTPUT_DIR / f"{DATASET_NAME}_confusion_matrix.png"
    )

    print("\n" + "=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)
    print(f"- {tuning_path}")
    print(f"- {test_metrics_path}")
    print(f"- {report_path}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_validation_curve_f1.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_train_val_gap.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_threshold_tuning.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
