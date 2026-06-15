"""Run all model pipelines from the project-level src directory."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(project_dir / "src" / "RunAll.py")]
    if args.force_train:
        command.append("--force-train")
    subprocess.run(command, cwd=project_dir, check=True)


if __name__ == "__main__":
    main()
