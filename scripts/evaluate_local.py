"""Local evaluation script for AIMO-3.

Runs the model on reference.csv (10 problems) and logs results to logs/experiments.csv.
Use this for fast PDCA iteration without pushing to Kaggle.

Usage:
    # Default config
    python scripts/evaluate_local.py

    # Override parameters
    python scripts/evaluate_local.py --temperature 0.3 --num_samples 4

    # Use a different model
    python scripts/evaluate_local.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

    # Quick test on first N problems
    python scripts/evaluate_local.py --max_problems 3
"""

import argparse
import csv
import gc
import json
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ExperimentConfig:
    model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    dtype: str = "bfloat16"
    temperature: float = 0.6
    top_p: float = 0.95
    num_samples: int = 8
    max_new_tokens: int = 16384
    max_model_len: int = 20480
    gpu_memory_utilization: float = 0.92
    seed: int = 42
    prompt_version: str = "v1"
    system_prompt: str = ""
    user_prompt_template: str = ""
    notes: str = ""


SYSTEM_PROMPT_V1 = """\
You are a world-class mathematician solving olympiad-level math problems.

Instructions:
- Think step by step with rigorous reasoning.
- Double-check your calculations.
- The answer is always a non-negative integer between 0 and 99999 inclusive.
- If the problem asks for a remainder modulo some number, compute it carefully.
- Present your final answer as: \\boxed{ANSWER} where ANSWER is a single integer.
"""

USER_PROMPT_V1 = """\
Solve the following math problem. Think step by step, verify your answer, then give your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer (0 to 99999).

Problem:
{problem}
"""

SYSTEM_PROMPT_TIR = """\
You are a world-class mathematician solving olympiad-level math problems.

Instructions:
- Think step by step with rigorous reasoning.
- You can write Python code to help with calculations. Put code in ```python blocks.
- After writing code, I will execute it and show you the output. Then continue reasoning.
- Use sympy for symbolic computation, exact arithmetic, and modular arithmetic when helpful.
- The answer is always a non-negative integer between 0 and 99999 inclusive.
- If the problem asks for a remainder modulo some number, compute it carefully.
- After all reasoning, present your final answer as: \\boxed{ANSWER} where ANSWER is a single integer.

Example of using code:
```python
from sympy import factorint, mod_inverse
result = pow(2, 1000, 99991)
print(result)
```
"""

USER_PROMPT_TIR = """\
Solve the following math problem. You may use Python code for calculations. Think step by step, verify your answer, then give your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer (0 to 99999).

Problem:
{problem}
"""

PROMPT_VERSIONS = {
    "v1": (SYSTEM_PROMPT_V1, USER_PROMPT_V1),
    "tir": (SYSTEM_PROMPT_TIR, USER_PROMPT_TIR),
}


def extract_answer(text: str) -> int | None:
    """Extract integer answer from model output using multiple patterns."""
    for m in reversed(re.findall(r"\\boxed\{(\d+)\}", text)):
        val = int(m)
        if 0 <= val <= 99999:
            return val
    for m in reversed(re.findall(r"(?:the\s+)?(?:answer|remainder|result)\s+is\s+(\d+)", text, re.I)):
        val = int(m)
        if 0 <= val <= 99999:
            return val
    for m in reversed(re.findall(r"=\s*(\d+)\s*$", text, re.M)):
        val = int(m)
        if 0 <= val <= 99999:
            return val
    for m in reversed(re.findall(r"\b(\d+)\b", text)):
        val = int(m)
        if 0 <= val <= 99999:
            return val
    return None


def majority_vote(answers: list[int | None]) -> int:
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0
    return Counter(valid).most_common(1)[0][0]


def load_reference(path: str = "data/raw/reference.csv") -> list[dict]:
    """Load reference problems with answers."""
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_dict("records")


def run_experiment(config: ExperimentConfig, problems: list[dict]) -> dict:
    """Run experiment and return results."""
    from vllm import LLM, SamplingParams

    system_prompt, user_template = PROMPT_VERSIONS.get(
        config.prompt_version, (config.system_prompt, config.user_prompt_template)
    )

    print(f"Loading model: {config.model}")
    t0 = time.time()
    llm = LLM(
        model=config.model,
        dtype=config.dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=True,
        seed=config.seed,
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        n=config.num_samples,
        seed=config.seed,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    results = []
    correct = 0
    total_time = 0

    for i, prob in enumerate(problems):
        pid = prob["id"]
        problem_text = prob["problem"]
        expected = int(prob["answer"])

        print(f"\n[{i+1}/{len(problems)}] Problem {pid} (expected: {expected})")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(problem=problem_text)},
        ]

        t1 = time.time()
        try:
            outputs = llm.chat([messages], sampling_params=sampling_params)
        except Exception:
            prompt = f"{system_prompt}\n\n{user_template.format(problem=problem_text)}"
            outputs = llm.generate([prompt], sampling_params=sampling_params)
        elapsed = time.time() - t1
        total_time += elapsed

        answers = []
        for output in outputs:
            for completion in output.outputs:
                answers.append(extract_answer(completion.text))

        valid = [a for a in answers if a is not None]
        predicted = majority_vote(answers)
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        print(f"  Predicted: {predicted} | Expected: {expected} | {'OK' if is_correct else 'WRONG'}")
        print(f"  Votes: {Counter(valid).most_common(3)} | Time: {elapsed:.1f}s")

        results.append({
            "id": pid,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "answers": dict(Counter(valid)),
            "extraction_rate": len(valid) / max(len(answers), 1),
            "time_seconds": round(elapsed, 1),
        })

        gc.collect()

    accuracy = correct / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0

    del llm
    gc.collect()

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_time_per_problem": round(avg_time, 1),
        "total_time": round(total_time, 1),
        "load_time": round(load_time, 1),
        "results": results,
    }


def log_experiment(config: ExperimentConfig, metrics: dict, log_path: str = "logs/experiments.csv"):
    """Append experiment results to CSV log."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "prompt_version": config.prompt_version,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_samples": config.num_samples,
        "max_new_tokens": config.max_new_tokens,
        "seed": config.seed,
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "avg_time_per_problem": metrics["avg_time_per_problem"],
        "total_time": metrics["total_time"],
        "load_time": metrics["load_time"],
        "notes": config.notes,
    }

    write_header = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # Also save detailed results
    detail_dir = Path("logs/details")
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / f"{experiment_id}.json"
    with open(detail_path, "w") as f:
        json.dump({"config": asdict(config), "metrics": metrics}, f, indent=2, default=str)

    print(f"\nExperiment logged: {experiment_id}")
    print(f"  Summary: {log_path}")
    print(f"  Details: {detail_path}")
    return experiment_id


def main():
    parser = argparse.ArgumentParser(description="Local evaluation for AIMO-3")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=20480)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_version", default="v1")
    parser.add_argument("--max_problems", type=int, default=None, help="Limit number of problems (for quick testing)")
    parser.add_argument("--tir", action="store_true", help="Enable Tool-Integrated Reasoning")
    parser.add_argument("--notes", default="", help="Notes for this experiment")
    parser.add_argument("--reference", default="data/raw/reference.csv", help="Path to reference CSV")
    args = parser.parse_args()

    config = ExperimentConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        prompt_version="tir" if args.tir else args.prompt_version,
        notes=args.notes + (" [TIR]" if args.tir else ""),
    )

    print("=" * 60)
    print("AIMO-3 Local Evaluation")
    print("=" * 60)
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}, Top-p: {config.top_p}")
    print(f"Samples: {config.num_samples}, Max tokens: {config.max_new_tokens}")
    print(f"Prompt version: {config.prompt_version}")
    print(f"Seed: {config.seed}")
    print()

    problems = load_reference(args.reference)
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"Evaluating on {len(problems)} problems\n")

    metrics = run_experiment(config, problems)

    print("\n" + "=" * 60)
    print(f"RESULTS: {metrics['correct']}/{metrics['total']} correct ({metrics['accuracy']:.1%})")
    print(f"Avg time per problem: {metrics['avg_time_per_problem']:.1f}s")
    print(f"Total time: {metrics['total_time']:.1f}s (+ {metrics['load_time']:.1f}s model load)")
    print("=" * 60)

    experiment_id = log_experiment(config, metrics)
    return experiment_id


if __name__ == "__main__":
    main()
