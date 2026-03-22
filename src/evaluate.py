"""Evaluation module for AIMO3.

Metrics:
- Simple accuracy: number of correct answers (public LB)
- Penalized accuracy: run twice, both correct=1, one correct=0.5, neither=0 (private LB)
"""

import csv
import json
from datetime import UTC, datetime

import numpy as np

from src.config import Config


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy (fraction of exact matches)."""
    return float(np.mean(y_true == y_pred))


def penalized_accuracy(y_true: np.ndarray, y_pred_run1: np.ndarray, y_pred_run2: np.ndarray) -> float:
    """Compute penalized accuracy score (private LB metric).

    For each problem:
    - Both runs correct: 1.0
    - One run correct: 0.5
    - Neither correct: 0.0
    """
    correct1 = y_true == y_pred_run1
    correct2 = y_true == y_pred_run2
    scores = correct1.astype(float) * 0.5 + correct2.astype(float) * 0.5
    return float(np.sum(scores))


def evaluate_reference(predictions: dict[str, int], reference_df) -> dict:
    """Evaluate predictions against reference problems.

    Args:
        predictions: dict of {id: predicted_answer}
        reference_df: DataFrame with 'id' and 'answer' columns

    Returns:
        dict with evaluation results
    """
    correct = 0
    total = 0
    details = []
    for _, row in reference_df.iterrows():
        pid = row["id"]
        true_answer = int(row["answer"])
        pred_answer = predictions.get(pid, 0)
        is_correct = pred_answer == true_answer
        correct += is_correct
        total += 1
        details.append({
            "id": pid,
            "true": true_answer,
            "pred": pred_answer,
            "correct": is_correct,
        })

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "details": details,
    }


def log_experiment(cfg: Config, result: dict) -> None:
    """Save experiment result as JSON and append to CSV in logs/."""
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    result["timestamp"] = timestamp

    json_path = cfg.logs_dir / f"{timestamp}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))

    csv_path = cfg.logs_dir / "experiments.csv"
    flat = {k: str(v) if isinstance(v, list | dict) else v for k, v in result.items()}
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)
