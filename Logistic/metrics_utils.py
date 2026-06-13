import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_proba, threshold):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def format_confusion_matrix(metrics):
    return f"[[{metrics['TN']}, {metrics['FP']}], [{metrics['FN']}, {metrics['TP']}]]"


def print_fold_metrics(fold, metrics):
    print(f"\nFold {fold}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"CM       : {format_confusion_matrix(metrics)}")


def print_cv_summary(summary):
    print("\n========== CV MEAN RESULT ==========")
    print(f"Accuracy : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"Recall   : {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    print(f"F1-score : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"ROC-AUC  : {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")
    print(f"CM total : {summary['confusion_matrix']}")


def print_survey_table(title, df):
    print("\n\n================================================")
    print(title)
    print("================================================")
    print(df.to_string(index=False))
    