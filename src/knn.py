from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from data_splitting import (
    stratified_sample,
    stratified_train_validation_test_split,
)
from metrics_utils import evaluate_scores, select_threshold
from result_reporting import save_explainability_artifacts, save_result_graphs

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

OUTPUT_DIR = REPO_ROOT / "knn_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

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


def to_numpy_float32(X):
    return X.values.astype(np.float32)


# =========================================================
# METRICS
# =========================================================

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

                best_threshold, threshold_df = select_threshold(
                    y_true=y_val_np,
                    y_score=y_val_prob,
                    thresholds=THRESHOLDS,
                )

                train_metrics, _ = evaluate_scores(
                    y_train_np, y_train_prob, best_threshold
                )

                val_metrics, _ = evaluate_scores(
                    y_val_np, y_val_prob, best_threshold
                )

                train_val_f1_gap = abs(
                    train_metrics["f1_class_1"] - val_metrics["f1_class_1"]
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
                    "train_precision_class_0": train_metrics["precision_class_0"],
                    "train_precision_class_1": train_metrics["precision_class_1"],
                    "train_recall_class_0": train_metrics["recall_class_0"],
                    "train_recall_class_1": train_metrics["recall_class_1"],
                    "train_f1_class_0": train_metrics["f1_class_0"],
                    "train_f1_class_1": train_metrics["f1_class_1"],
                    "train_roc_auc": train_metrics["roc_auc"],

                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision_class_0": val_metrics["precision_class_0"],
                    "val_precision_class_1": val_metrics["precision_class_1"],
                    "val_recall_class_0": val_metrics["recall_class_0"],
                    "val_recall_class_1": val_metrics["recall_class_1"],
                    "val_f1_class_0": val_metrics["f1_class_0"],
                    "val_f1_class_1": val_metrics["f1_class_1"],
                    "val_roc_auc": val_metrics["roc_auc"],

                    "train_val_f1_gap": train_val_f1_gap,
                }

                tuning_rows.append(row)
                threshold_store[config_id] = threshold_df

                print(
                    f"Config {config_id:02d} | "
                    f"k={k}, weights={weights}, metric={metric_name} | "
                    f"val_f1={val_metrics['f1_class_1']:.4f}, "
                    f"val_recall={val_metrics['recall_class_1']:.4f}, "
                    f"val_auc={val_metrics['roc_auc']:.4f}, "
                    f"gap={train_val_f1_gap:.4f}"
                )

    tuning_df = pd.DataFrame(tuning_rows)

    tuning_df = tuning_df.sort_values(
        [
            "val_f1_class_1",
            "val_recall_class_1",
            "val_roc_auc",
            "train_val_f1_gap",
        ],
        ascending=[False, False, False, True]
    )

    best_config_id = int(tuning_df.iloc[0]["config_id"])

    return tuning_df, threshold_store, best_config_id


# =========================================================
# FINAL TEST
# =========================================================

def final_evaluate_best_model(best_row, X_train, y_train, X_test, y_test):
    X_final_train, y_final_train = stratified_sample(
        X_train,
        y_train,
        MAX_FINAL_TRAIN_ROWS,
        RANDOM_STATE,
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
    test_metrics, y_test_pred = evaluate_scores(
        y_test_np, y_test_prob, threshold
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
        y="val_f1_class_1",
        hue="setting",
        marker="o"
    )

    plt.title("KNN validation F1-score by k")
    plt.xlabel("Number of neighbors k")
    plt.ylabel("Validation F1-score for Class 1")
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
        plot_df["precision_class_1"],
        marker="o",
        label="Precision Class 1"
    )
    plt.plot(
        plot_df["threshold"],
        plot_df["recall_class_1"],
        marker="o",
        label="Recall Class 1"
    )
    plt.plot(
        plot_df["threshold"],
        plot_df["f1_class_1"],
        marker="o",
        label="F1 Class 1"
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
        xticklabels=["Pred Class 0", "Pred Class 1"],
        yticklabels=["True Class 0", "True Class 1"]
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

    split = stratified_train_validation_test_split(X, y, RANDOM_STATE)
    X_train, X_val, X_test, y_train, y_val, y_test = split

    print("\nData split:")
    print(f"- Train:      {X_train.shape}")
    print(f"- Validation: {X_val.shape}")
    print(f"- Test:       {X_test.shape}")

    print("\nClass distribution:")
    print(f"- Train Class 1 ratio:      {y_train.mean():.4f}")
    print(f"- Validation Class 1 ratio: {y_val.mean():.4f}")
    print(f"- Test Class 1 ratio:       {y_test.mean():.4f}")

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
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_path = OUTPUT_DIR / f"{DATASET_NAME}_test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("FINAL TEST METRICS")
    print("=" * 100)
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

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
    save_result_graphs(
        y_test_np,
        y_test_prob,
        y_test_pred,
        test_metrics,
        OUTPUT_DIR / "result_graphs",
        "KNN",
    )
    save_explainability_artifacts(
        model_name="KNN",
        output_dir=OUTPUT_DIR / "explainability",
        X_background=X_train,
        X_explain=X_test,
        feature_names=feature_names,
        predict_proba_fn=model.predict_proba,
    )

    print("\n" + "=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)
    print(f"- {tuning_path}")
    print(f"- {test_metrics_path}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_validation_curve_f1.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_train_val_gap.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_threshold_tuning.png'}")
    print(f"- {OUTPUT_DIR / f'{DATASET_NAME}_confusion_matrix.png'}")
    print(f"- {OUTPUT_DIR / 'result_graphs'}")
    print(f"- {OUTPUT_DIR / 'explainability'}")


if __name__ == "__main__":
    main()
