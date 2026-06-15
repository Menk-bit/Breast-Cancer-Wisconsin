"""K-fold tune, train, save, and evaluate a Support Vector Machine."""

import argparse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ModelUtils import run_experiment
from Preprocessing import RANDOM_STATE, prepare_data


PARAMETER_GRID = [
    {
        "model__kernel": ["linear"],
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1.0, 10.0, 100.0],
        "model__gamma": ["scale", 0.01, 0.1],
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__kernel": ["poly"],
        "model__C": [0.1, 1.0, 10.0],
        "model__degree": [2, 3],
        "model__gamma": ["scale"],
        "model__class_weight": [None, "balanced"],
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    run_experiment(
        model_name="Support Vector Machine",
        artifact_name="svm",
        estimator=estimator,
        parameter_grid=PARAMETER_GRID,
        split=prepare_data(),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
