"""K-fold tune, train, save, and evaluate a Random Forest."""

import argparse

from sklearn.ensemble import RandomForestClassifier

from ModelUtils import run_experiment
from Preprocessing import RANDOM_STATE, prepare_data


# Candidate hyperparameters; selection uses 5-fold CV on training data only.
PARAMETER_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "class_weight": [None, "balanced"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    run_experiment(
        model_name="Random Forest",
        artifact_name="random_forest",
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        parameter_grid=PARAMETER_GRID,
        split=prepare_data(),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
