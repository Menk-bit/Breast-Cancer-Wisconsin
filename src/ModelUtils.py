"""Shared K-fold tuning, persistence, evaluation, and artifact generation."""

from __future__ import annotations

import json
import hashlib
import time
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
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold

from Preprocessing import CV_SPLITS, RANDOM_STATE, DataSplit


PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def select_best_index(cv_results: dict[str, Any]) -> int:
    """Prioritize survivor recall, then F1, ROC-AUC, and accuracy."""
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
    signature_path = model_path.with_suffix(".signature")
    model_contract = {
        "dataset": split.dataset_signature,
        "estimator": type(estimator).__name__,
        "parameters": estimator.get_params(),
        "grid": parameter_grid,
    }
    model_signature = hashlib.sha256(
        json.dumps(model_contract, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if model_path.exists() and signature_path.exists() and not force_train:
        saved_signature = signature_path.read_text(encoding="utf-8").strip()
        if saved_signature == model_signature:
            print("[1/4] Loading compatible saved model...", flush=True)
            model = joblib.load(model_path)
            print("[2/4] Saved model ready.", flush=True)
            return model, False

    candidate_count = len(ParameterGrid(parameter_grid))
    fit_count = candidate_count * CV_SPLITS
    print(
        f"[1/4] Training: {candidate_count} candidates x "
        f"{CV_SPLITS} folds = {fit_count} CV fits",
        flush=True,
    )
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
        verbose=1,
    )
    search_started = time.perf_counter()
    search.fit(split.X_train, split.y_train)
    print(
        f"[2/4] Cross-validation complete in "
        f"{format_duration(time.perf_counter() - search_started)}",
        flush=True,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_path)
    signature_path.write_text(model_signature, encoding="utf-8")
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
            "survival_score": float(score),
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
            target_names=["Did not survive 5 years", "Survived 5 years"],
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
        display_labels=["No", "Yes"],
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
    false_negative_ids = [item["id"] for item in metrics["false_negatives"]]
    print(f"{'FN rows':>10}: {len(false_negative_ids)}")
    print(f"{'First 20':>10}: {false_negative_ids[:20]}")


def run_experiment(
    model_name: str,
    artifact_name: str,
    estimator: Any,
    parameter_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
    split: DataSplit,
    force_train: bool,
) -> dict[str, Any]:
    experiment_started = time.perf_counter()
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
    print("[3/4] Evaluating the held-out test set...", flush=True)
    metrics = evaluate_model(model, model_name, split)
    print("[4/4] Saving metrics and result plot...", flush=True)
    save_results(model, metrics, split, output_dir)
    print_summary(metrics, model_path, trained)
    print(
        f"Experiment finished in "
        f"{format_duration(time.perf_counter() - experiment_started)}.",
        flush=True,
    )
    return metrics
