#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS = SCRIPT_DIR / "experiments.json"
DEFAULT_FLWR = SCRIPT_DIR.parent / ".venv" / "bin" / "flwr"


def load_experiments(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("experiments", data) if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError("Expected a non-empty list of experiments")
    return [(float(i["target_epsilon"]), float(i["dirichlet_alpha"])) for i in items]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiments from experiments.json")
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--flwr-bin", type=Path, default=DEFAULT_FLWR)
    args = parser.parse_args()

    try:
        experiments = load_experiments(args.experiments)
        flwr = str(args.flwr_bin if args.flwr_bin.exists() else (shutil.which("flwr") or ""))
        if not flwr:
            raise FileNotFoundError("flwr executable not found")

        env = os.environ.copy()
        env["PATH"] = f"{Path(flwr).resolve().parent}:{env.get('PATH', '')}"

        for idx, (eps, alpha) in enumerate(experiments, start=1):
            print(f"\n[{idx}/{len(experiments)}] eps={eps}, alpha={alpha}")
            subprocess.run(
                [flwr, "run", ".", "--stream", "--run-config", f"target-epsilon={eps} dirichlet-alpha={alpha}"],
                cwd=SCRIPT_DIR,
                env=env,
                check=True,
            )
    except (FileNotFoundError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
