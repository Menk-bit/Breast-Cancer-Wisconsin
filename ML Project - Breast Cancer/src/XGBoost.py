"""K-fold tune, train, save, and evaluate XGBoost."""

import argparse
import sys
from pathlib import Path

from ModelUtils import run_experiment
from Preprocessing import RANDOM_STATE, prepare_data


# Windows imports are case-insensitive, so this file can shadow `xgboost`.
SCRIPT_DIR = Path(__file__).resolve().parent
original_path = sys.path.copy()
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != SCRIPT_DIR
]
try:
    from xgboost import XGBClassifier
finally:
    sys.path = original_path


# Candidate hyperparameters; selection uses 5-fold CV on training data only.
PARAMETER_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.08],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8],
    "min_child_weight": [1, 3],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    run_experiment(
        model_name="XGBoost",
        artifact_name="xgboost",
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        parameter_grid=PARAMETER_GRID,
        split=prepare_data("tree"),
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
