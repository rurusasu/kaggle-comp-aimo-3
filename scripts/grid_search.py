"""Grid search over hyperparameters using evaluate_local.py.

Usage:
    # Run full grid search
    python scripts/grid_search.py

    # Quick test (1 problem per config)
    python scripts/grid_search.py --max_problems 1
"""

import argparse
import itertools
import subprocess
import sys


GRID = {
    "temperature": [0.0, 0.3, 0.6],
    "num_samples": [4, 8],
    "prompt_version": ["v1"],
}


def main():
    parser = argparse.ArgumentParser(description="Grid search for AIMO-3")
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    args = parser.parse_args()

    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID.keys())

    print(f"Grid search: {len(combos)} configurations")
    print(f"Parameters: {keys}")
    print()

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        note = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(combos)}] {note}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "scripts/evaluate_local.py",
            "--model", args.model,
            "--notes", f"grid_search: {note}",
        ]
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
        if args.max_problems:
            cmd.extend(["--max_problems", str(args.max_problems)])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {e}")
            continue

    print(f"\nGrid search complete. Check logs/experiments.csv for results.")


if __name__ == "__main__":
    main()
