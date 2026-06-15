import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def format_confusion_matrix(metrics):
    return f"[[{metrics['TN']}, {metrics['FP']}], [{metrics['FN']}, {metrics['TP']}]]"


def summarize_fold_metrics(fold_metrics):
    summary = {}

    for metric in ["accuracy", "recall", "f1"]:
        values = np.array([item[metric] for item in fold_metrics], dtype=float)
        summary[f"{metric}_mean"] = np.mean(values)
        summary[f"{metric}_std"] = np.std(values)

    tn = sum(item["TN"] for item in fold_metrics)
    fp = sum(item["FP"] for item in fold_metrics)
    fn = sum(item["FN"] for item in fold_metrics)
    tp = sum(item["TP"] for item in fold_metrics)

    summary["TN"] = tn
    summary["FP"] = fp
    summary["FN"] = fn
    summary["TP"] = tp
    summary["confusion_matrix"] = f"[[{tn}, {fp}], [{fn}, {tp}]]"

    return summary


def sort_cv_results(cv_results):
    df = pd.DataFrame(cv_results)

    return df.sort_values(
        by=[
            "recall_mean",
            "FN",
            "f1_mean",
            "accuracy_mean",
            "FP",
        ],
        ascending=[
            False,
            True,
            False,
            False,
            True,
        ],
    )


def select_best_config(cv_results):
    sorted_df = sort_cv_results(cv_results)
    return sorted_df.iloc[0].to_dict(), sorted_df


def build_display_table(cv_results, top_n):
    sorted_df = sort_cv_results(cv_results).head(top_n)

    rows = []

    for _, row in sorted_df.iterrows():
        corr_value = row["corr_threshold"]

        if pd.isna(corr_value):
            corr_display = "-"
        else:
            corr_display = corr_value

        rows.append(
            {
                "baseline": row["baseline"],
                "k": int(row["k"]),
                "features": row["feature_mode"],
                "corr": corr_display,
                "scale": row["use_scaling"],
                "distance": row["distance_metric"],
                "weights": row["weights"],
                "n_features": f"{row['n_features_mean']:.1f}",
                "Accuracy": f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}",
                "Recall": f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f}",
                "F1-score": f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f}",
                "CM [TN,FP;FN,TP]": row["confusion_matrix"],
            }
        )

    return pd.DataFrame(rows)


def print_baseline_cv_summary(baseline_name, cv_results, top_n):
    display_df = build_display_table(cv_results, top_n)

    print("\n\n================================================")
    print(f"CV SUMMARY - {baseline_name}")
    print("================================================")
    print(f"Top {top_n} configs by medical priority")
    print(display_df.to_string(index=False))


def print_global_cv_summary(cv_results, top_n):
    display_df = build_display_table(cv_results, top_n)

    print("\n\n================================================")
    print("GLOBAL CV SUMMARY")
    print("================================================")
    print(f"Top {top_n} configs across enabled baselines")
    print(display_df.to_string(index=False))


def print_best_cv_config(best_config):
    print("\n\n================================================")
    print("BEST CV CONFIG")
    print("================================================")
    print(f"baseline      : {best_config['baseline']}")
    print(f"k             : {int(best_config['k'])}")
    print(f"feature_mode  : {best_config['feature_mode']}")
    print(f"corr_threshold: {best_config['corr_threshold']}")
    print(f"use_scaling   : {best_config['use_scaling']}")
    print(f"distance      : {best_config['distance_metric']}")
    print(f"weights       : {best_config['weights']}")
    print(f"n_features    : {best_config['n_features_mean']:.1f}")

    print("\nCV result:")
    print(f"Accuracy      : {best_config['accuracy_mean']:.4f} ± {best_config['accuracy_std']:.4f}")
    print(f"Recall        : {best_config['recall_mean']:.4f} ± {best_config['recall_std']:.4f}")
    print(f"F1-score      : {best_config['f1_mean']:.4f} ± {best_config['f1_std']:.4f}")
    print(f"CM total      : {best_config['confusion_matrix']}")


def print_test_result(metrics):
    print("\n\n================================================")
    print("FINAL TEST RESULT")
    print("================================================")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"CM       : {format_confusion_matrix(metrics)}")