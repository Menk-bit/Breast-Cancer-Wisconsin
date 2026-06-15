"""Run all model tuning and evaluation pipelines."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS = [
    "KNN.py",
    "LogisticRegression.py",
    "LinearRegression.py",
    "NaiveBayes.py",
    "SVM.py",
    "RandomForest.py",
    "XGBoost.py",
]


def progress_bar(completed: int, total: int, width: int = 28) -> str:
    filled = round(width * completed / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total}"


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    total_models = len(SCRIPTS)
    run_started = time.perf_counter()

    print(f"Running {total_models} model pipelines", flush=True)
    print(progress_bar(0, total_models), flush=True)

    for index, script in enumerate(SCRIPTS, start=1):
        command = [sys.executable, str(source_dir / script)]
        if args.force_train:
            command.append("--force-train")
        model_started = time.perf_counter()
        print(
            f"\n{'=' * 70}\n"
            f"{progress_bar(index - 1, total_models)} Starting {script}\n"
            f"{'=' * 70}",
            flush=True,
        )
        subprocess.run(command, cwd=source_dir, check=True)
        elapsed = time.perf_counter() - model_started
        print(
            f"{progress_bar(index, total_models)} "
            f"Completed {script} in {format_duration(elapsed)}",
            flush=True,
        )

    total_elapsed = time.perf_counter() - run_started
    print(
        f"\nAll model pipelines completed in {format_duration(total_elapsed)}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
