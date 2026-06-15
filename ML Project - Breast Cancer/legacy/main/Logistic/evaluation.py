import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from metrics_utils import calculate_metrics, format_confusion_matrix
from model import LogisticRegressionScratch


def train_final_model(
    X_train,
    y_train,
    C,
    learning_rate,
    class_weight,
    epochs,
    print_every,
):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)

    model = LogisticRegressionScratch(
        learning_rate=learning_rate,
        epochs=epochs,
        C=C,
        class_weight=class_weight,
        print_every=print_every,
        verbose=False,
    )

    model.fit(X_train_scaled, y_train)

    return model, imputer, scaler


def evaluate_on_test(
    model,
    imputer,
    scaler,
    X_test,
    y_test,
    threshold,
):
    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)

    y_test_proba = model.predict_proba(X_test_scaled)

    return calculate_metrics(
        y_true=y_test,
        y_proba=y_test_proba,
        threshold=threshold,
    )


def print_test_result(title, metrics):
    print("\n\n================================================")
    print(title)
    print("================================================")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"CM       : {format_confusion_matrix(metrics)}")