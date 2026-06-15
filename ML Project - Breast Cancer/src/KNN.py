"""K-fold tune, train, save, and evaluate K-nearest neighbors."""

import argparse

from sklearn.neighbors import KNeighborsClassifier

from ModelUtils import run_experiment
from Preprocessing import prepare_data


PARAMETER_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["manhattan", "euclidean"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    run_experiment(
        model_name="K-Nearest Neighbors",
        artifact_name="knn",
        estimator=KNeighborsClassifier(n_jobs=-1),
        parameter_grid=PARAMETER_GRID,
        split=prepare_data("scaled"),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
