"""K-fold tune, train, save, and evaluate a linear SVM computed by SGD."""

import argparse

from sklearn.linear_model import SGDClassifier

from ModelUtils import run_experiment
from Preprocessing import RANDOM_STATE, prepare_data


PARAMETER_GRID = [
    {
        "alpha": [1e-5, 5e-5, 1e-4, 5e-4],
        "class_weight": [None, "balanced"],
        "learning_rate": ["optimal"],
        "max_iter": [15, 20],
    },
    {
        "alpha": [1e-5, 5e-5, 1e-4, 5e-4],
        "class_weight": [None, "balanced"],
        "learning_rate": ["invscaling"],
        "eta0": [0.02, 0.05, 0.08],
        "power_t": [0.5],
        "max_iter": [15, 20],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    run_experiment(
        model_name="Support Vector Machine (SGD)",
        artifact_name="svm",
        estimator=SGDClassifier(
            loss="hinge",
            penalty="l2",
            tol=None,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        parameter_grid=PARAMETER_GRID,
        split=prepare_data("scaled"),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
