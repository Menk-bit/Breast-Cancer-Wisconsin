"""Gradient-descent logistic classifier adapted from LR + NB/v03.ipynb."""

import argparse

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ModelUtils import run_experiment
from Preprocessing import RANDOM_STATE, prepare_data


class GradientDescentLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Binary logistic regression trained with mini-batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 50,
        batch_size: int = 4096,
        l2: float = 0.0,
        random_state: int = RANDOM_STATE,
    ):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_array.shape[1]
        self.coef_ = np.zeros((1, self.n_features_in_), dtype=np.float64)
        self.intercept_ = np.zeros(1, dtype=np.float64)
        self.loss_history_ = []
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.max_epochs):
            for indices in np.array_split(
                rng.permutation(len(X_array)),
                max(1, int(np.ceil(len(X_array) / self.batch_size))),
            ):
                X_batch = X_array[indices]
                y_batch = y_array[indices]
                probabilities = self._sigmoid(
                    X_batch @ self.coef_[0] + self.intercept_[0]
                )
                error = probabilities - y_batch
                gradient = X_batch.T @ error / len(indices)
                gradient += self.l2 * self.coef_[0]
                self.coef_[0] -= self.learning_rate * gradient
                self.intercept_[0] -= self.learning_rate * error.mean()

            probabilities = self._sigmoid(
                X_array @ self.coef_[0] + self.intercept_[0]
            )
            probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
            loss = -np.mean(
                y_array * np.log(probabilities)
                + (1 - y_array) * np.log(1 - probabilities)
            )
            loss += 0.5 * self.l2 * np.sum(self.coef_[0] ** 2)
            self.loss_history_.append(float(loss))
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X) -> np.ndarray:
        positive = self._sigmoid(self.decision_function(X))
        return np.column_stack((1 - positive, positive))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


PARAMETER_GRID = {
    "learning_rate": [0.005, 0.01],
    "max_epochs": [30, 60],
    "l2": [0.0, 0.001],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    run_experiment(
        model_name="Gradient-Descent Logistic Regression",
        artifact_name="linear_regression",
        estimator=GradientDescentLogisticClassifier(),
        parameter_grid=PARAMETER_GRID,
        split=prepare_data("scaled"),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
