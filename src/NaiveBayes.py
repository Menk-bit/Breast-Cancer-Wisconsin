"""Gaussian Naive Bayes adapted from LR + NB/v03.ipynb."""

import argparse

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ModelUtils import run_experiment
from Preprocessing import prepare_data


class NotebookGaussianNB(ClassifierMixin, BaseEstimator):
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=np.float64)
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        self.n_features_in_ = X_array.shape[1]
        self.class_prior_ = np.array(
            [np.mean(y_array == label) for label in self.classes_]
        )
        self.theta_ = np.vstack(
            [X_array[y_array == label].mean(axis=0) for label in self.classes_]
        )
        variances = np.vstack(
            [X_array[y_array == label].var(axis=0) for label in self.classes_]
        )
        epsilon = self.var_smoothing * max(float(X_array.var()), 1.0)
        self.var_ = variances + epsilon
        return self

    def predict_log_proba(self, X) -> np.ndarray:
        X_array = np.asarray(X, dtype=np.float64)
        log_posteriors = []
        for index in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[index])
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[index])
                + ((X_array - self.theta_[index]) ** 2) / self.var_[index],
                axis=1,
            )
            log_posteriors.append(log_prior + log_likelihood)
        values = np.column_stack(log_posteriors)
        normalizer = np.logaddexp.reduce(values, axis=1)
        return values - normalizer[:, None]

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        indices = np.argmax(self.predict_log_proba(X), axis=1)
        return self.classes_[indices]


PARAMETER_GRID = {
    "var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    run_experiment(
        model_name="Gaussian Naive Bayes",
        artifact_name="naive_bayes",
        estimator=NotebookGaussianNB(),
        parameter_grid=PARAMETER_GRID,
        split=prepare_data("scaled"),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
