"""AIMO3 Submission — DeepSeek-R1-0528-Qwen3-8B + vLLM + TIR + Majority Voting.

TIR (Tool-Integrated Reasoning): model generates Python code blocks, vLLM stops
at ```output, code is executed, result appended, then generation continues.
Based on the Numina (AIMO-1 winner) SC-TIR approach.

Key: stop=["```output\\n"] allows batched multi-turn code execution with vLLM.
"""

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# vLLM installation
# ---------------------------------------------------------------------------
try:
    import vllm  # noqa: F401
    print(f"vllm {vllm.__version__} already installed")
except ImportError:
    candidates = [
        "/kaggle/input/vllm-0-13-0-with-dependencies-for-offline-install/wheels",
        "/kaggle/input/vllm-0-13-0-with-dependencies-for-offline-install",
        "/kaggle/input/datasets/denizyunusg/vllm-0-13-0-with-dependencies-for-offline-install/wheels",
        "/kaggle/input/datasets/denizyunusg/vllm-0-13-0-with-dependencies-for-offline-install",
    ]
    wheels_dir = None
    for c in candidates:
        if os.path.isdir(c):
            wheels_dir = c
            break
    if wheels_dir:
        import glob as _glob
        whl_files = _glob.glob(os.path.join(wheels_dir, "*.whl"))
        skip = ("torch-", "nvidia_", "triton-", "numpy-", "sympy-", "networkx-")
        filtered = [w for w in whl_files if not any(os.path.basename(w).startswith(p) for p in skip)]
        print(f"Installing vllm from {len(filtered)} wheels...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + filtered)
    else:
        print("No offline wheels — installing vllm online...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "vllm"])

import gc
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SAMPLES = 8              # parallel candidates per problem
NUM_TIR_ROUNDS = 4           # max code execution rounds
MAX_NEW_TOKENS = 2048        # per generation step (shorter, multiple rounds)
TEMPERATURE = 0.8            # higher diversity for SC-TIR
TOP_P = 0.95
GPU_MEMORY_UTILIZATION = 0.92
MAX_MODEL_LEN = 16384        # total context window
SEED = 42
TIME_BUDGET = 17400          # 5h - 10min buffer
PROBLEM_COUNT = 110
CODE_TIMEOUT = 10            # seconds per code execution

SAFE_IMPORTS = """\
import math
import itertools
import functools
from fractions import Fraction
from collections import Counter, defaultdict
import sympy as sp
from sympy import *
try:
    import numpy as np
except ImportError:
    pass
"""

SYSTEM_PROMPT = """\
You are a world-class mathematician. Solve the problem step by step.
You can write Python code in ```python blocks. After each code block, write ```output to see execution results.
Use sympy for symbolic computation. The final answer is a non-negative integer (0-99999).
End with \\boxed{ANSWER}.
"""

USER_PROMPT_TEMPLATE = """\
Solve this math problem step by step. Use Python code when helpful for calculations.

Problem:
{problem}

Solution:"""


# ---------------------------------------------------------------------------
# Code execution via subprocess (safe, with timeout)
# ---------------------------------------------------------------------------
def execute_code(code: str, timeout: int = CODE_TIMEOUT) -> str:
    """Execute Python code in a subprocess. Returns output string (max 200 chars)."""
    blocked = ["subprocess", "venv", "shutil", "os.system", "__import__", "open("]
    if any(b in code for b in blocked):
        return "Error: blocked operation"

    full_code = SAFE_IMPORTS + "\n" + code
    # Auto-print last expression if no print statement
    lines = full_code.strip().split("\n")
    if lines and not any("print" in l for l in lines[-3:]):
        last = lines[-1].strip()
        if last and not last.startswith(("#", "import", "from", "def", "class", "if", "for", "while", "try", "with")):
            lines[-1] = f"print({last})"
            full_code = "\n".join(lines)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp_path = f.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
            output = f"Error: {error}"
    except subprocess.TimeoutExpired:
        output = f"Error: timeout after {timeout}s"
    except Exception as e:
        output = f"Error: {e}"
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass

    return output[:200]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> int | None:
    for m in reversed(re.findall(r"\\boxed\{(\d+)\}", text)):
        val = int(m)
        if 0 <= val <= 99999:
            return val
    for m in reversed(re.findall(r"(?:the\s+)?(?:answer|remainder|result)\s+is\s+(\d+)", text, re.I)):
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def find_model_path(base="/kaggle/input"):
    for root, _dirs, files in os.walk(base):
        if "config.json" in files and "tokenizer.json" in files:
            return root
    for root, _dirs, files in os.walk(base):
        if "config.json" in files:
            return root
    return None


# ---------------------------------------------------------------------------
# SC-TIR: batched multi-turn code execution
# ---------------------------------------------------------------------------
def process_code_in_samples(gen_texts: list[str]) -> list[str]:
    """For each sample, if it ends with ```output\\n, execute the last code block
    and append the result. Otherwise, leave unchanged."""
    results = []
    for text in gen_texts:
        if text.rstrip().endswith("```output"):
            # Extract last Python code block
            code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
            if code_blocks:
                code = code_blocks[-1].strip()
                output = execute_code(code)
                text = text + "\n" + output + "\n```\n"
            else:
                text = text + "\nNo code found\n```\n"
        results.append(text)
    return results


def sc_tir_predict(llm, tokenizer, problem: str, num_samples: int, num_rounds: int) -> int:
    """Self-Consistency with Tool-Integrated Reasoning.

    Generate num_samples candidates in parallel, each going through up to
    num_rounds of code generation + execution.
    """
    from vllm import SamplingParams

    # Build initial prompt
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(problem=problem)}"

    # Try chat template first
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(problem=problem)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass

    # Initialize all samples with the same prompt
    gen_texts = [prompt] * num_samples

    sp = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        n=1,
        stop=["```output\n", "```output"],
        include_stop_str_in_output=True,
        seed=SEED,
    )

    for round_idx in range(num_rounds):
        # Generate for all samples in batch
        outputs = llm.generate(gen_texts, sampling_params=sp, use_tqdm=False)

        # Append generated text to accumulated text
        new_gen_texts = []
        for i, output in enumerate(outputs):
            new_text = output.outputs[0].text
            new_gen_texts.append(gen_texts[i] + new_text)

        gen_texts = new_gen_texts

        # Check how many samples need code execution
        needs_exec = sum(1 for t in gen_texts if t.rstrip().endswith(("```output", "```output\n")))
        if needs_exec == 0:
            break

        print(f"    Round {round_idx+1}: {needs_exec}/{num_samples} samples need code execution")

        # Execute code and append results
        gen_texts = process_code_in_samples(gen_texts)

    # Extract answers from all samples
    answers = []
    for text in gen_texts:
        answer = extract_answer(text)
        answers.append(answer)

    valid = [a for a in answers if a is not None]
    result = majority_vote(answers)
    print(f"  Answers: {Counter(valid).most_common(3)} → {result}")
    return result


# ---------------------------------------------------------------------------
# Model class with lazy loading
# ---------------------------------------------------------------------------
class Model:
    def __init__(self):
        self._llm = None
        self._tokenizer = None
        self._start_time = None
        self._problem_idx = 0

    def load(self):
        from vllm import LLM

        print("Loading model...")
        t0 = time.time()

        model_path = find_model_path() or "/kaggle/input/deepseek-r1-0528/transformers/deepseek-r1-0528-qwen3-8b/1"
        print(f"Model path: {model_path}")

        self._llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            seed=SEED,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._start_time = time.time()
        print(f"Model loaded in {self._start_time - t0:.1f}s")

    def predict(self, problem: str) -> int:
        if self._llm is None:
            self.load()

        self._problem_idx += 1
        elapsed = time.time() - self._start_time
        remaining = TIME_BUDGET - elapsed
        problems_left = max(PROBLEM_COUNT - self._problem_idx + 1, 1)
        per_problem = min(remaining / problems_left, 300)

        print(f"  [{self._problem_idx}/{PROBLEM_COUNT}] elapsed={elapsed:.0f}s remaining={remaining:.0f}s budget={per_problem:.0f}s")

        if remaining < 30:
            return 0

        try:
            result = sc_tir_predict(
                self._llm, self._tokenizer, problem,
                num_samples=NUM_SAMPLES, num_rounds=NUM_TIR_ROUNDS,
            )
        except Exception as e:
            print(f"  TIR failed: {e}, falling back to direct generation")
            result = self._direct_predict(problem)

        gc.collect()
        return result

    def _direct_predict(self, problem: str) -> int:
        """Fallback: single-shot generation without TIR."""
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=TEMPERATURE, top_p=TOP_P,
            max_tokens=4096, n=NUM_SAMPLES, seed=SEED,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(problem=problem)},
        ]
        try:
            outputs = self._llm.chat([messages], sampling_params=sp)
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(problem=problem)}"
            outputs = self._llm.generate([prompt], sampling_params=sp)

        answers = []
        for output in outputs:
            for comp in output.outputs:
                answers.append(extract_answer(comp.text))
        return majority_vote(answers)


model = Model()


# ---------------------------------------------------------------------------
# Predict function for kaggle_evaluation API
# ---------------------------------------------------------------------------
def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    id_val = id_.item(0)
    problem_text: str = problem.item(0)
    print(f"\n{'='*60}\nProblem {id_val}: {problem_text[:150]}...")

    try:
        answer = model.predict(problem_text)
    except Exception as e:
        print(f"  ERROR: {e}")
        answer = 0

    return pl.DataFrame({"id": id_val, "answer": answer})


# ---------------------------------------------------------------------------
# Inference server
# ---------------------------------------------------------------------------
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    test_candidates = [
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",
        "/kaggle/input/competitions/ai-mathematical-olympiad-progress-prize-3/test.csv",
    ]
    test_path = next((p for p in test_candidates if os.path.exists(p)), None)
    if test_path:
        inference_server.run_local_gateway((test_path,))
    else:
        print("No test.csv — skipping local gateway. Use 'Submit to Competition' in Kaggle.")
