"""K-fold tune, train, save, and evaluate Logistic Regression."""

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ModelUtils import run_experiment
from Preprocessing import RANDOM_STATE, prepare_data


PARAMETER_GRID = {
    "model__solver": ["liblinear", "lbfgs"],
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    run_experiment(
        model_name="Logistic Regression",
        artifact_name="logistic_regression",
        estimator=estimator,
        parameter_grid=PARAMETER_GRID,
        split=prepare_data(),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
