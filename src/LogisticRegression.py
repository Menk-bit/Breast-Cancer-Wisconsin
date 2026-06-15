"""K-fold tune, train, save, and evaluate Logistic Regression."""

import argparse

from sklearn.linear_model import LogisticRegression

from ModelUtils import run_experiment
from Preprocessing import RANDOM_STATE, prepare_data


PARAMETER_GRID = {
    "solver": ["liblinear", "lbfgs"],
    "C": [0.01, 0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    run_experiment(
        model_name="Logistic Regression",
        artifact_name="logistic_regression",
        estimator=LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        parameter_grid=PARAMETER_GRID,
        split=prepare_data("scaled"),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
