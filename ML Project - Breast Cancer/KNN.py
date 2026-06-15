"""K-fold tune, train, save, and evaluate K-nearest neighbors."""

import argparse

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ModelUtils import run_experiment
from Preprocessing import prepare_data


PARAMETER_GRID = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )
    run_experiment(
        model_name="K-Nearest Neighbors",
        artifact_name="knn",
        estimator=estimator,
        parameter_grid=PARAMETER_GRID,
        split=prepare_data(),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
