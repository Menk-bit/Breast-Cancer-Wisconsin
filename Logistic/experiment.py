import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from metrics_utils import (
    calculate_metrics,
    print_cv_summary,
    print_fold_metrics,
)
from model import LogisticRegressionScratch


def preprocess_fold_data(X_train_fold, X_valid_fold):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_fold)
    X_valid_imp = imputer.transform(X_valid_fold)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_valid_scaled = scaler.transform(X_valid_imp)

    return X_train_scaled, X_valid_scaled


def summarize_fold_results(fold_results):
    summary = {}

    for key in ["accuracy", "recall", "f1", "roc_auc"]:
        values = np.array([result[key] for result in fold_results], dtype=float)

        summary[f"{key}_mean"] = np.nanmean(values)
        summary[f"{key}_std"] = np.nanstd(values)

    tn = sum(result["TN"] for result in fold_results)
    fp = sum(result["FP"] for result in fold_results)
    fn = sum(result["FN"] for result in fold_results)
    tp = sum(result["TP"] for result in fold_results)

    summary.update(
        {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "confusion_matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
        }
    )

    return summary


def run_cv_for_config(
    X,
    y,
    C,
    learning_rate,
    class_weight,
    threshold,
    n_splits,
    random_state,
    epochs,
    print_every,
    print_each_fold=False,
    keep_history=False,
):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_results = []
    first_history = None

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]

        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]

        X_train_scaled, X_valid_scaled = preprocess_fold_data(
            X_train_fold,
            X_valid_fold,
        )

        model = LogisticRegressionScratch(
            learning_rate=learning_rate,
            epochs=epochs,
            C=C,
            class_weight=class_weight,
            print_every=print_every,
            verbose=False,
        )

        model.fit(
            X_train_scaled,
            y_train_fold,
            X_valid=X_valid_scaled,
            y_valid=y_valid_fold,
        )

        y_valid_proba = model.predict_proba(X_valid_scaled)

        metrics = calculate_metrics(
            y_valid_fold,
            y_valid_proba,
            threshold,
        )

        fold_results.append(metrics)

        if print_each_fold:
            print_fold_metrics(fold, metrics)

        if keep_history and first_history is None:
            first_history = model.get_history()

    summary = summarize_fold_results(fold_results)

    return fold_results, summary, first_history


def run_baseline_report(
    X,
    y,
    C,
    learning_rate,
    class_weight,
    threshold,
    n_splits,
    random_state,
    epochs,
    print_every,
):
    print("\n\n================================================")
    print("BASELINE CONFIG")
    print("================================================")
    print(f"C            : {C}")
    print(f"learning_rate: {learning_rate}")
    print(f"class_weight : {class_weight}")
    print(f"threshold    : {threshold}")

    fold_results, summary, history = run_cv_for_config(
        X=X,
        y=y,
        C=C,
        learning_rate=learning_rate,
        class_weight=class_weight,
        threshold=threshold,
        n_splits=n_splits,
        random_state=random_state,
        epochs=epochs,
        print_every=print_every,
        print_each_fold=True,
        keep_history=True,
    )

    print_cv_summary(summary)

    return fold_results, summary, history


def format_summary_row(parameter_name, parameter_value, summary):
    value = "None" if parameter_value is None else parameter_value

    return {
        parameter_name: value,
        "Accuracy": f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}",
        "Recall": f"{summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}",
        "F1-score": f"{summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}",
        "ROC-AUC": f"{summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}",
        "CM [TN,FP;FN,TP]": summary["confusion_matrix"],
    }


def run_parameter_survey(
    X,
    y,
    parameter_name,
    parameter_values,
    base_config,
    n_splits,
    random_state,
    epochs,
    print_every,
):
    rows = []

    for value in parameter_values:
        config = base_config.copy()
        config[parameter_name] = value

        _, summary, _ = run_cv_for_config(
            X=X,
            y=y,
            C=config["C"],
            learning_rate=config["learning_rate"],
            class_weight=config["class_weight"],
            threshold=config["threshold"],
            n_splits=n_splits,
            random_state=random_state,
            epochs=epochs,
            print_every=print_every,
            print_each_fold=False,
            keep_history=False,
        )

        rows.append(
            format_summary_row(
                parameter_name,
                value,
                summary,
            )
        )

    return pd.DataFrame(rows)


def run_c_survey(
    X,
    y,
    C_values,
    base_config,
    n_splits,
    random_state,
    epochs,
    print_every,
):
    return run_parameter_survey(
        X=X,
        y=y,
        parameter_name="C",
        parameter_values=C_values,
        base_config=base_config,
        n_splits=n_splits,
        random_state=random_state,
        epochs=epochs,
        print_every=print_every,
    )


def run_learning_rate_survey(
    X,
    y,
    learning_rate_values,
    base_config,
    n_splits,
    random_state,
    epochs,
    print_every,
):
    return run_parameter_survey(
        X=X,
        y=y,
        parameter_name="learning_rate",
        parameter_values=learning_rate_values,
        base_config=base_config,
        n_splits=n_splits,
        random_state=random_state,
        epochs=epochs,
        print_every=print_every,
    )


def run_class_weight_survey(
    X,
    y,
    class_weight_values,
    base_config,
    n_splits,
    random_state,
    epochs,
    print_every,
):
    return run_parameter_survey(
        X=X,
        y=y,
        parameter_name="class_weight",
        parameter_values=class_weight_values,
        base_config=base_config,
        n_splits=n_splits,
        random_state=random_state,
        epochs=epochs,
        print_every=print_every,
    )


def run_threshold_survey(
    X,
    y,
    threshold_values,
    base_config,
    n_splits,
    random_state,
    epochs,
    print_every,
):
    return run_parameter_survey(
        X=X,
        y=y,
        parameter_name="threshold",
        parameter_values=threshold_values,
        base_config=base_config,
        n_splits=n_splits,
        random_state=random_state,
        epochs=epochs,
        print_every=print_every,
    )
