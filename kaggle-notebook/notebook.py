"""AIMO3 Baseline Submission Notebook (Kaggle Code Competition).

This is the template for the Kaggle notebook submission.
Copy this code into a Kaggle notebook with:
- GPU T4x2 or H100 accelerator
- Internet disabled
- Model weights attached as a Kaggle dataset

Approach:
- Load a strong open-source math reasoning model (e.g., DeepSeek-R1-Qwen3-8B)
- For each problem served by the API:
  1. Generate N solutions with chain-of-thought reasoning
  2. Extract integer answer from each solution
  3. Use majority voting to select final answer
- Submit via kaggle_evaluation API
"""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "vllm"])

import gc
import os
import re
import time
from collections import Counter

# -------------------------------------------------------------------
# Debug: show available inputs
# -------------------------------------------------------------------
print("=== /kaggle/input contents ===")
for root, dirs, files in os.walk("/kaggle/input"):
    level = root.replace("/kaggle/input", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        subindent = " " * 2 * (level + 1)
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more")

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
# Auto-detect model path
def find_config_json(base="/kaggle/input"):
    for root, dirs, files in os.walk(base):
        if "config.json" in files and "tokenizer.json" in files:
            return root
    for root, dirs, files in os.walk(base):
        if "config.json" in files:
            return root
    return None

MODEL_PATH = find_config_json() or "/kaggle/input/deepseek-r1-0528/transformers/deepseek-r1-0528-qwen3-8b/1"
print(f"Using model: {MODEL_PATH}")
MAX_NEW_TOKENS = 32768
TEMPERATURE = 0.6
TOP_P = 0.95
NUM_SAMPLES = 16  # majority voting samples
GPU_MEMORY_UTILIZATION = 0.90
TIME_BUDGET = 17400  # 5h - 10min buffer
TIME_PER_PROBLEM_MAX = 1500  # 25 min max per problem

SYSTEM_PROMPT = """\
You are an expert mathematician solving olympiad-level math problems.

Rules:
- Think step by step, showing all reasoning.
- The answer is always a non-negative integer between 0 and 99999 inclusive.
- Any modular arithmetic required is explicitly stated in the problem.
- After your reasoning, provide your final answer as: \\boxed{ANSWER}
- ANSWER must be a single non-negative integer.
"""

USER_PROMPT_TEMPLATE = """\
Solve the following math problem. Show your work step by step, then give your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer (0 to 99999).

Problem:
{problem}
"""

# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------


def extract_answer(text: str) -> int | None:
    """Extract integer answer from model output."""
    boxed_matches = re.findall(r"\\boxed\{(\d+)\}", text)
    if boxed_matches:
        val = int(boxed_matches[-1])
        if 0 <= val <= 99999:
            return val
    int_matches = re.findall(r"\b(\d+)\b", text)
    if int_matches:
        val = int(int_matches[-1])
        if 0 <= val <= 99999:
            return val
    return None


def majority_vote(answers: list[int | None]) -> int:
    """Select answer by majority vote."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0
    return Counter(valid).most_common(1)[0][0]


# -------------------------------------------------------------------
# 3. Load model
# -------------------------------------------------------------------
print("Loading model...")
start_time = time.time()

from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_NEW_TOKENS + 4096,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_NEW_TOKENS,
    n=NUM_SAMPLES,
)

print(f"Model loaded in {time.time() - start_time:.1f}s")


# -------------------------------------------------------------------
# 4. Solve function
# -------------------------------------------------------------------
def solve(problem: str) -> int:
    """Solve a single math problem."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(problem=problem)},
    ]

    try:
        outputs = llm.chat([messages], sampling_params=sampling_params)
    except Exception:
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(problem=problem)}"
        outputs = llm.generate([prompt], sampling_params=sampling_params)

    answers = []
    for output in outputs:
        for completion in output.outputs:
            answer = extract_answer(completion.text)
            answers.append(answer)
            print(f"  Sample answer: {answer}")

    result = majority_vote(answers)
    print(f"  Majority vote: {result}")
    return result


# -------------------------------------------------------------------
# 5. Kaggle API submission loop
# -------------------------------------------------------------------
import kaggle_evaluation.aimo_3_submission

competition_start = time.time()


def predict(id_: str, problem: str) -> int | None:
    """Callback for kaggle_evaluation API."""
    elapsed = time.time() - competition_start
    remaining = TIME_BUDGET - elapsed
    print(f"\n--- Problem {id_} (elapsed: {elapsed:.0f}s, remaining: {remaining:.0f}s) ---")
    print(f"Problem: {problem[:200]}...")

    if remaining < 60:
        print("  Time running out, returning 0")
        return 0

    try:
        answer = solve(problem)
    except Exception as e:
        print(f"  Error: {e}")
        answer = 0

    gc.collect()
    return answer


# Register and run
inference_server = kaggle_evaluation.aimo_3_submission.AIMOSubmission(predict)
inference_server.serve()
