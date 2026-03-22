"""Inference entrypoint. Runs solver on reference problems (requires GPU + vLLM).

Usage:
    uv run python scripts/predict.py
    uv run python scripts/predict.py --num-samples 8 --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_reference
from src.evaluate import evaluate_reference, log_experiment
from src.model import load_vllm_model, solve_problem
from src.submit import create_submission
from src.utils import Timer, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, num_samples=args.num_samples)
    if args.model:
        cfg.model_name = args.model
    set_seed(cfg.seed)

    with Timer("load reference"):
        ref_df = load_reference(cfg)

    with Timer("load model"):
        llm = load_vllm_model(cfg)

    predictions = {}
    for idx, row in ref_df.iterrows():
        pid = row["id"]
        problem = row["problem"]
        print(f"\n--- Problem {idx + 1}/{len(ref_df)}: {pid} ---")
        print(f"  True answer: {row['answer']}")

        with Timer(f"solve {pid}"):
            pred = solve_problem(llm, problem, cfg)

        predictions[pid] = pred
        print(f"  Predicted: {pred}, Correct: {pred == row['answer']}")

    # Evaluate
    result = evaluate_reference(predictions, ref_df)
    print(f"\n=== Results ===")
    print(f"Accuracy: {result['correct']}/{result['total']} = {result['accuracy']:.2%}")

    # Save submission
    ids = [d["id"] for d in result["details"]]
    preds = [d["pred"] for d in result["details"]]
    sub_path = create_submission(cfg, ids, preds)
    print(f"Submission saved to {sub_path}")

    # Log
    log_experiment(cfg, {
        "experiment": f"baseline_{cfg.model_name}",
        "model": cfg.model_name,
        "num_samples": cfg.num_samples,
        "accuracy": result["accuracy"],
        "correct": result["correct"],
        "total": result["total"],
    })


if __name__ == "__main__":
    main()
