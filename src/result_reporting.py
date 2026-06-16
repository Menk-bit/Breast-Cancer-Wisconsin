"""Shared result plots plus SHAP and LIME explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from metrics_utils import METRIC_COLUMNS


PredictProba = Callable[[np.ndarray], np.ndarray]


def _as_frame(data, feature_names: list[str] | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    names = feature_names or [f"feature_{index}" for index in range(np.asarray(data).shape[1])]
    return pd.DataFrame(np.asarray(data), columns=names)


def _two_column_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    return probabilities


def save_result_graphs(
    y_true,
    y_score,
    y_pred,
    metrics: dict[str, float],
    output_dir: Path,
    model_name: str,
) -> None:
    """Save standard result graphs for a binary classifier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred, dtype=int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{model_name}: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Class 0", "Class 1"])
    plt.yticks([0, 1], ["Class 0", "Class 1"])
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(f"{model_name}: ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.title(f"{model_name}: Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    metric_values = {column: metrics[column] for column in METRIC_COLUMNS if column in metrics}
    plt.figure(figsize=(11, 5))
    plt.bar(metric_values.keys(), metric_values.values())
    plt.ylim(0, 1.05)
    plt.title(f"{model_name}: Standard Metrics")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_explainability_artifacts(
    *,
    model_name: str,
    output_dir: Path,
    X_background,
    X_explain,
    feature_names: list[str],
    predict_proba_fn: PredictProba,
    model=None,
    shap_sample_size: int = 200,
    background_sample_size: int = 200,
    lime_instances: int = 3,
) -> None:
    """Save compact SHAP and LIME artifacts for any tabular binary classifier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    background = _as_frame(X_background, feature_names).sample(
        n=min(background_sample_size, len(X_background)),
        random_state=42,
    )
    explain = _as_frame(X_explain, feature_names).sample(
        n=min(shap_sample_size, len(X_explain)),
        random_state=43,
    )

    _save_shap(
        model_name=model_name,
        output_dir=output_dir,
        background=background,
        explain=explain,
        predict_proba_fn=predict_proba_fn,
        model=model,
    )
    _save_lime(
        model_name=model_name,
        output_dir=output_dir,
        background=background,
        explain=explain,
        feature_names=feature_names,
        predict_proba_fn=predict_proba_fn,
        lime_instances=lime_instances,
    )


def _save_shap(
    *,
    model_name: str,
    output_dir: Path,
    background: pd.DataFrame,
    explain: pd.DataFrame,
    predict_proba_fn: PredictProba,
    model=None,
) -> None:
    try:
        import shap

        if model is not None and hasattr(model, "feature_importances_"):
            explainer = shap.Explainer(model, background)
            explanation = explainer(explain)
            values = explanation.values
            if values.ndim == 3:
                values = values[:, :, 1]
        else:
            explainer = shap.KernelExplainer(
                lambda rows: _two_column_probabilities(predict_proba_fn(rows))[:, 1],
                background,
            )
            values = explainer.shap_values(explain, silent=True)
            if isinstance(values, list):
                values = values[-1]

        values = np.asarray(values, dtype=float)
        mean_abs = pd.DataFrame(
            {
                "feature": explain.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        mean_abs.to_csv(output_dir / "shap_feature_importance.csv", index=False)

        plt.figure()
        shap.summary_plot(values, explain, show=False, max_display=20)
        plt.title(f"{model_name}: SHAP Summary")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # noqa: BLE001 - keep model training from failing.
        (output_dir / "shap_unavailable.txt").write_text(str(exc), encoding="utf-8")


def _save_lime(
    *,
    model_name: str,
    output_dir: Path,
    background: pd.DataFrame,
    explain: pd.DataFrame,
    feature_names: list[str],
    predict_proba_fn: PredictProba,
    lime_instances: int,
) -> None:
    try:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            training_data=background.to_numpy(dtype=float),
            feature_names=feature_names,
            class_names=["class_0", "class_1"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        def predict(rows: np.ndarray) -> np.ndarray:
            return _two_column_probabilities(predict_proba_fn(rows))

        rows = []
        for index in range(min(lime_instances, len(explain))):
            explanation = explainer.explain_instance(
                explain.iloc[index].to_numpy(dtype=float),
                predict,
                num_features=min(15, len(feature_names)),
                labels=[1],
            )
            explanation.save_to_file(output_dir / f"lime_explanation_{index}.html")
            for feature, weight in explanation.as_list(label=1):
                rows.append({"instance": index, "feature_rule": feature, "weight": weight})

        pd.DataFrame(rows).to_csv(output_dir / "lime_explanations.csv", index=False)
    except Exception as exc:  # noqa: BLE001 - keep model training from failing.
        (output_dir / "lime_unavailable.txt").write_text(str(exc), encoding="utf-8")
