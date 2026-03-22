"""AIMO3 Baseline Submission Notebook (Kaggle Code Competition).

Uses kaggle_evaluation.aimo_3_inference_server API with lazy model loading.
Model is loaded on first predict() call so that inference_server.serve()
can be called within the 15-minute startup deadline.

Approach:
- DeepSeek-R1-0528-Qwen3-8B via vLLM
- Generate N solutions with chain-of-thought reasoning
- Extract integer answer from each solution
- Majority voting to select final answer
"""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "vllm"])

import gc
import os
import re
import time
from collections import Counter

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MAX_NEW_TOKENS = 32768
TEMPERATURE = 0.6
TOP_P = 0.95
NUM_SAMPLES = 16
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
# Helper functions
# -------------------------------------------------------------------
def find_config_json(base="/kaggle/input"):
    """Auto-detect model path by looking for config.json."""
    for root, dirs, files in os.walk(base):
        if "config.json" in files and "tokenizer.json" in files:
            return root
    for root, dirs, files in os.walk(base):
        if "config.json" in files:
            return root
    return None


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
# Model class with lazy loading
# -------------------------------------------------------------------
class Model:
    """Wraps vLLM model with lazy loading."""

    def __init__(self):
        self._llm = None
        self._sampling_params = None
        self._start_time = None

    def load(self):
        """Load the vLLM model. Called on first predict() call."""
        from vllm import LLM, SamplingParams

        print("Loading model...")
        load_start = time.time()

        model_path = find_config_json() or "/kaggle/input/deepseek-r1-0528/transformers/deepseek-r1-0528-qwen3-8b/1"
        print(f"Using model: {model_path}")

        self._llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_NEW_TOKENS + 4096,
            trust_remote_code=True,
        )
        self._sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_NEW_TOKENS,
            n=NUM_SAMPLES,
        )
        self._start_time = time.time()
        print(f"Model loaded in {self._start_time - load_start:.1f}s")

    def predict(self, problem: str) -> int:
        """Solve a single math problem with majority voting."""
        # Lazy load on first call
        if self._llm is None:
            self.load()

        elapsed = time.time() - self._start_time
        remaining = TIME_BUDGET - elapsed
        print(f"  (elapsed: {elapsed:.0f}s, remaining: {remaining:.0f}s)")

        if remaining < 60:
            print("  Time running out, returning 0")
            return 0

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(problem=problem)},
        ]

        try:
            outputs = self._llm.chat([messages], sampling_params=self._sampling_params)
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(problem=problem)}"
            outputs = self._llm.generate([prompt], sampling_params=self._sampling_params)

        answers = []
        for output in outputs:
            for completion in output.outputs:
                answer = extract_answer(completion.text)
                answers.append(answer)
                print(f"  Sample answer: {answer}")

        result = majority_vote(answers)
        print(f"  Majority vote: {result}")

        gc.collect()
        return result


model = Model()


# -------------------------------------------------------------------
# Predict function for kaggle_evaluation API
# -------------------------------------------------------------------
def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction for a single problem."""
    id_val = id_.item(0)
    problem_text: str = problem.item(0)

    print(f"\n--- Problem {id_val} ---")
    print(f"Problem: {problem_text[:200]}...")

    try:
        answer = model.predict(problem_text)
    except Exception as e:
        print(f"  Error: {e}")
        answer = 0

    return pl.DataFrame({"id": id_val, "answer": answer})


# -------------------------------------------------------------------
# Inference server setup
# -------------------------------------------------------------------
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ("/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",)
    )
