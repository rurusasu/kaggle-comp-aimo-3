"""Local evaluation against reference problems.

Usage:
    uv run python scripts/train.py

This tests the solver against the 10 reference problems with known answers.
Not a traditional training script - this competition uses pre-trained LLMs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_reference
from src.evaluate import evaluate_reference, log_experiment
from src.model import extract_answer, majority_vote, build_prompt, SYSTEM_PROMPT
from src.features import classify_problem_domain
from src.utils import Timer, set_seed


def main():
    cfg = Config()
    set_seed(cfg.seed)

    with Timer("load reference"):
        ref_df = load_reference(cfg)

    print(f"Loaded {len(ref_df)} reference problems\n")

    # Show problem domains
    for _, row in ref_df.iterrows():
        domain = classify_problem_domain(row["problem"])
        print(f"  {row['id']}: {domain} (answer={row['answer']})")

    print("\n--- To run full evaluation, use scripts/predict.py with a GPU ---")
    print("--- This script validates the pipeline without GPU inference ---\n")

    # Test answer extraction
    test_cases = [
        ("The answer is \\boxed{42}", 42),
        ("Therefore \\boxed{99999}", 99999),
        ("Computing, we get 336.", 336),
        ("No valid answer here.", None),
    ]
    print("Testing answer extraction:")
    for text, expected in test_cases:
        result = extract_answer(text, cfg)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] extract_answer({text!r}) = {result} (expected {expected})")

    # Test majority voting
    print("\nTesting majority voting:")
    test_votes = [
        ([42, 42, 42, 7, None], 42),
        ([None, None, None], 0),
        ([1, 2, 2, 3, 3, 3], 3),
    ]
    for votes, expected in test_votes:
        result = majority_vote(votes)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] majority_vote({votes}) = {result} (expected {expected})")


if __name__ == "__main__":
    main()
