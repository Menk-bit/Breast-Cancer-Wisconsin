"""Train and compare Random Forest and XGBoost on the local SEER dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data_splitting import (
    stratified_sample,
    stratified_train_validation_test_split,
)
from metrics_utils import (
    METRIC_COLUMNS,
    evaluate_scores,
    select_threshold_for_class_0,
)
from result_reporting import save_explainability_artifacts, save_result_graphs

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "model_ready_tree.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tree_ensemble_outputs"
TARGET_COLUMN = "survive_after_5"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Random Forest and XGBoost on model_ready_tree.csv."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for a faster run.",
    )
    parser.add_argument(
        "--max-configs-per-model",
        type=int,
        default=None,
        help="Optional cap for faster hyperparameter search smoke runs.",
    )
    parser.add_argument(
        "--explainability-sample-size",
        type=int,
        default=200,
        help="Maximum rows used for SHAP and LIME background/explanation samples.",
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip SHAP and LIME artifact generation.",
    )
    return parser.parse_args()


def load_data(
    path: Path, sample_size: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    df = df.loc[df[TARGET_COLUMN].notna()].copy()
    y = pd.to_numeric(df.pop(TARGET_COLUMN), errors="raise").astype(np.int8)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1, found: {sorted(y.unique())}")

    leakage_columns = [
        column
        for column in (
            "event_dead",
            "survival_months",
            "survival_months_int",
            "survival_months_unknown_flag",
        )
        if column in df.columns
    ]
    X = df.drop(columns=leakage_columns)
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)

    if sample_size is not None and sample_size < len(X):
        if sample_size < 100:
            raise ValueError("--sample-size must be at least 100.")
        X, y = stratified_sample(X, y, sample_size, RANDOM_STATE)

    return X.reset_index(drop=True), y.reset_index(drop=True)


def random_forest_grid() -> list[dict[str, object]]:
    return list(
        ParameterGrid(
            {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [16],
                "min_samples_leaf": [5, 10],
                "max_features": ["sqrt"],
                "class_weight": ["balanced_subsample"],
            }
        )
    )


def xgboost_grid() -> list[dict[str, object]]:
    return list(
        ParameterGrid(
            {
                "n_estimators": [200, 400, 600, 800],
                "max_depth": [5, 7],
                "learning_rate": [0.05],
                "subsample": [0.85],
                "colsample_bytree": [0.85],
                "min_child_weight": [3],
                "reg_alpha": [0.05],
                "reg_lambda": [1.0],
            }
        )
    )


def build_model(name: str, params: dict[str, object]) -> object:
    if name == "random_forest":
        return RandomForestClassifier(
            **params,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {name}")


def parameter_grids() -> dict[str, list[dict[str, object]]]:
    return {
        "random_forest": random_forest_grid(),
        "xgboost": xgboost_grid(),
    }


def fit_model(
    name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
) -> float:
    start = time.perf_counter()
    if name == "xgboost":
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return time.perf_counter() - start


def save_model_outputs(
    name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    metrics: dict[str, float | int],
    output_dir: Path,
) -> None:
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "probability_survive_after_5": probabilities,
            "predicted": predictions,
        }
    ).to_csv(model_dir / "test_predictions.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)

    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_result_graphs(
        y_test,
        probabilities,
        predictions,
        metrics,
        model_dir,
        name,
    )


def run_hyperparameter_search(
    name: str,
    grid: list[dict[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    sample_weight: np.ndarray,
    output_dir: Path,
    max_configs: int | None,
) -> tuple[object, pd.Series, pd.DataFrame, pd.DataFrame]:
    rows = []
    best_model = None
    best_row = None
    best_threshold_frame = None
    configs = grid[:max_configs] if max_configs else grid

    for config_id, params in enumerate(configs, start=1):
        model = build_model(name, params)
        training_seconds = fit_model(name, model, X_train, y_train, sample_weight)
        validation_probabilities = model.predict_proba(X_valid)[:, 1]
        threshold, threshold_frame = select_threshold_for_class_0(
            y_valid,
            validation_probabilities,
        )
        validation_metrics, _ = evaluate_scores(
            y_valid,
            validation_probabilities,
            threshold,
        )
        row = {
            "config_id": config_id,
            **params,
            **validation_metrics,
            "training_seconds": training_seconds,
        }
        rows.append(row)

        is_best = best_row is None or (
            row["recall_class_0"],
            row["f1_class_0"],
            row["roc_auc"],
        ) > (
            best_row["recall_class_0"],
            best_row["f1_class_0"],
            best_row["roc_auc"],
        )
        if is_best:
            best_model = model
            best_row = row
            best_threshold_frame = threshold_frame

        print(
            f"{name} config {config_id:02d}/{len(configs)} | "
            f"recall_0={validation_metrics['recall_class_0']:.4f}, "
            f"f1_0={validation_metrics['f1_class_0']:.4f}, "
            f"roc_auc={validation_metrics['roc_auc']:.4f}, "
            f"threshold={threshold:.3f}, time={training_seconds:.1f}s"
        )

    if best_model is None or best_row is None or best_threshold_frame is None:
        raise RuntimeError(f"No hyperparameter configs were evaluated for {name}.")

    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    search_frame = pd.DataFrame(rows).sort_values(
        ["recall_class_0", "f1_class_0", "roc_auc"],
        ascending=False,
    )
    search_frame.to_csv(model_dir / "hyperparameter_search.csv", index=False)
    best_threshold_frame.to_csv(
        model_dir / "validation_threshold_metrics.csv",
        index=False,
    )
    plot_hyperparameter_curves(name, search_frame, model_dir)
    plot_threshold_curve(best_threshold_frame, model_dir)
    return best_model, pd.Series(best_row), search_frame, best_threshold_frame


def plot_hyperparameter_curves(
    name: str,
    search_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    estimator_column = "n_estimators"
    if estimator_column not in search_frame.columns:
        return

    grouped = (
        search_frame.groupby(estimator_column)[
            ["recall_class_0", "f1_class_0", "roc_auc"]
        ]
        .max()
        .reset_index()
        .sort_values(estimator_column)
    )
    plt.figure(figsize=(9, 5))
    for metric in ["recall_class_0", "f1_class_0", "roc_auc"]:
        plt.plot(grouped[estimator_column], grouped[metric], marker="o", label=metric)
    plt.title(f"{name}: Hyperparameter Curve")
    plt.xlabel("n_estimators")
    plt.ylabel("Validation score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparameter_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_threshold_curve(threshold_frame: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for metric in ["recall_class_0", "f1_class_0", "precision_class_0"]:
        plt.plot(threshold_frame["threshold"], threshold_frame[metric], label=metric)
    plt.title("Validation Threshold Curve")
    plt.xlabel("Threshold")
    plt.ylabel("Validation score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data.resolve(), args.sample_size)
    split = stratified_train_validation_test_split(X, y, RANDOM_STATE)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split
    print(
        f"Loaded {len(X):,} rows and {X.shape[1]} features "
        f"(positive rate: {y.mean():.3f})."
    )
    print(
        f"Split sizes: train={len(X_train):,}, validation={len(X_valid):,}, "
        f"test={len(X_test):,}."
    )

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    summary_rows = []

    for name, grid in parameter_grids().items():
        print(f"\nSearching {name} hyperparameters...")
        model, best_validation_row, _, _ = run_hyperparameter_search(
            name=name,
            grid=grid,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            sample_weight=sample_weight,
            output_dir=output_dir,
            max_configs=args.max_configs_per_model,
        )

        threshold = float(best_validation_row["threshold"])
        test_probabilities = model.predict_proba(X_test)[:, 1]
        metrics, predictions = evaluate_scores(
            y_test, test_probabilities, threshold
        )
        metrics["training_seconds"] = float(best_validation_row["training_seconds"])
        metrics["rows"] = len(X)
        metrics["features"] = X.shape[1]
        metrics["selected_validation_recall_class_0"] = float(
            best_validation_row["recall_class_0"]
        )
        metrics["selected_validation_f1_class_0"] = float(
            best_validation_row["f1_class_0"]
        )

        save_model_outputs(
            name,
            model,
            X_test,
            y_test,
            test_probabilities,
            predictions,
            metrics,
            output_dir,
        )
        best_validation_row.to_frame().T.to_csv(
            output_dir / name / "selected_hyperparameters.csv",
            index=False,
        )
        if not args.skip_explainability:
            save_explainability_artifacts(
                model_name=name,
                output_dir=output_dir / name / "explainability",
                X_background=X_train,
                X_explain=X_test,
                feature_names=X_train.columns.tolist(),
                predict_proba_fn=model.predict_proba,
                model=model,
                shap_sample_size=args.explainability_sample_size,
                background_sample_size=args.explainability_sample_size,
            )
        summary_rows.append({"model": name, **metrics})
        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"recall_class_0={metrics['recall_class_0']:.4f}, "
            f"f1_class_0={metrics['f1_class_0']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"threshold={threshold:.3f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["recall_class_0", "f1_class_0", "roc_auc"], ascending=False
    )
    summary.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\nSaved results to {output_dir}")
    print(
        summary[
            [
                "model",
                *METRIC_COLUMNS,
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
