"""Show experiment results from logs/experiments.csv.

Usage:
    python scripts/show_experiments.py
"""

import csv
from pathlib import Path


def main():
    log_path = Path("logs/experiments.csv")
    if not log_path.exists():
        print("No experiments found. Run scripts/evaluate_local.py first.")
        return

    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No experiments found.")
        return

    # Sort by accuracy descending
    rows.sort(key=lambda r: float(r.get("accuracy", 0)), reverse=True)

    print(f"{'ID':<22} {'Accuracy':>8} {'Correct':>7} {'T(°C)':>5} {'N':>3} {'Prompt':>6} {'Time/P':>7} {'Notes'}")
    print("-" * 100)
    for r in rows:
        acc = float(r.get("accuracy", 0))
        print(
            f"{r['experiment_id']:<22} "
            f"{acc:>7.1%} "
            f"{r['correct']:>3}/{r['total']:<3} "
            f"{r['temperature']:>5} "
            f"{r['num_samples']:>3} "
            f"{r['prompt_version']:>6} "
            f"{r['avg_time_per_problem']:>6}s "
            f"{r.get('notes', '')}"
        )

    print(f"\nTotal experiments: {len(rows)}")
    best = rows[0]
    print(f"Best: {best['experiment_id']} — {float(best['accuracy']):.1%} ({best['correct']}/{best['total']})")


if __name__ == "__main__":
    main()
