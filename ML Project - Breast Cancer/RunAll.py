"""Compatibility entry point for the model pipelines now stored in src."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    command = [sys.executable, str(root_dir / "src" / "RunAll.py")]
    if args.force_train:
        command.append("--force-train")
    subprocess.run(command, cwd=root_dir, check=True)


if __name__ == "__main__":
    main()
