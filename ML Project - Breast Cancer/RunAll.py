"""Run all model tuning and evaluation pipelines."""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "KNN.py",
    "LogisticRegression.py",
    "SVM.py",
    "RandomForest.py",
    "XGBoost.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent

    for script in SCRIPTS:
        command = [sys.executable, str(project_dir / script)]
        if args.force_train:
            command.append("--force-train")
        print(f"\n{'=' * 70}\nRunning {script}\n{'=' * 70}", flush=True)
        subprocess.run(command, cwd=project_dir, check=True)


if __name__ == "__main__":
    main()
