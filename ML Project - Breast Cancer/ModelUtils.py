"""Shared K-fold tuning, persistence, evaluation, and plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from Preprocessing import CV_SPLITS, RANDOM_STATE, DataSplit


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def select_best_index(cv_results: dict[str, Any]) -> int:
    """Prioritize malignant recall, then F1, ROC-AUC, and accuracy."""
    results = pd.DataFrame(cv_results)
    ranked = results.sort_values(
        by=[
            "mean_test_recall",
            "mean_test_f1",
            "mean_test_roc_auc",
            "mean_test_accuracy",
        ],
        ascending=False,
    )
    return int(ranked.index[0])


def train_or_load(
    estimator: Any,
    parameter_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
    split: DataSplit,
    model_path: Path,
    tuning_path: Path,
    force_train: bool,
) -> tuple[Any, bool]:
    if model_path.exists() and not force_train:
        return joblib.load(model_path), False

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring=SCORING,
        refit=select_best_index,
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(split.X_train, split.y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_path)
    tuning_results = pd.DataFrame(search.cv_results_).sort_values(
        by=[
            "mean_test_recall",
            "mean_test_f1",
            "mean_test_roc_auc",
            "mean_test_accuracy",
        ],
        ascending=False,
    )
    tuning_results.to_csv(tuning_path, index=False)
    return search.best_estimator_, True


def evaluate_model(model: Any, model_name: str, split: DataSplit) -> dict[str, Any]:
    y_pred = model.predict(split.X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(split.X_test)[:, 1]
    else:
        y_score = model.decision_function(split.X_test)

    tn, fp, fn, tp = confusion_matrix(
        split.y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()
    false_negatives = [
        {
            "id": int(sample_id),
            "malignant_score": float(score),
        }
        for sample_id, score, actual, predicted in zip(
            split.test_ids,
            y_score,
            split.y_test,
            y_pred,
        )
        if actual == 1 and predicted == 0
    ]
    return {
        "model": model_name,
        "accuracy": accuracy_score(split.y_test, y_pred),
        "precision": precision_score(split.y_test, y_pred, zero_division=0),
        "recall": recall_score(split.y_test, y_pred, zero_division=0),
        "f1": f1_score(split.y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(split.y_test, y_score),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "false_negatives": false_negatives,
        "classification_report": classification_report(
            split.y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
            output_dict=True,
        ),
        "best_hyperparameters": model.get_params(),
        "_y_pred": y_pred,
        "_y_score": y_score,
    }


def save_results(
    model: Any,
    metrics: dict[str, Any],
    split: DataSplit,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value for key, value in metrics.items() if not key.startswith("_")
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2, default=str),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        split.y_test,
        metrics["_y_pred"],
        display_labels=["Benign", "Malignant"],
        cmap="Blues",
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    RocCurveDisplay.from_predictions(
        split.y_test,
        metrics["_y_score"],
        name=metrics["model"],
        ax=axes[1],
    )
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_title("ROC Curve")

    importances = get_feature_importances(model, split.X_test.columns)
    if importances is None:
        axes[2].axis("off")
        axes[2].text(
            0.5,
            0.5,
            "Feature importance is not available\nfor this estimator.",
            ha="center",
            va="center",
        )
    else:
        importances.nlargest(12).sort_values().plot.barh(
            ax=axes[2],
            color="#4472C4",
        )
        axes[2].set_title("Top 12 Feature Importances")
        axes[2].set_xlabel("Importance")

    figure.suptitle(f"{metrics['model']} Test Results", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "results.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def get_feature_importances(model: Any, feature_names: Any) -> pd.Series | None:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.steps[-1][1]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
    else:
        return None
    return pd.Series(values, index=feature_names)


def print_summary(metrics: dict[str, Any], model_path: Path, trained: bool) -> None:
    print(f"Model: {'K-fold tuned and saved' if trained else 'loaded'} ({model_path})")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{key:>10}: {metrics[key]:.4f}")
    print(
        f"{'CM':>10}: [[{metrics['TN']}, {metrics['FP']}], "
        f"[{metrics['FN']}, {metrics['TP']}]]"
    )
    print(
        "False-negative IDs:",
        [item["id"] for item in metrics["false_negatives"]],
    )


def run_experiment(
    model_name: str,
    artifact_name: str,
    estimator: Any,
    parameter_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
    split: DataSplit,
    force_train: bool,
) -> dict[str, Any]:
    output_dir = ARTIFACTS_DIR / artifact_name
    model_path = output_dir / f"{artifact_name}.joblib"
    model, trained = train_or_load(
        estimator,
        parameter_grid,
        split,
        model_path,
        output_dir / "cv_results.csv",
        force_train,
    )
    metrics = evaluate_model(model, model_name, split)
    save_results(model, metrics, split, output_dir)
    print_summary(metrics, model_path, trained)
    return metrics
