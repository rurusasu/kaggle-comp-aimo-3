"""SC-TIR: Self-Consistency with Tool-Integrated Reasoning.

Based on Numina (AIMO-1 winner) approach:
- vLLM stops at ```output\\n
- Code is executed, result appended
- All samples batched in parallel
"""

import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

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


def execute_code(code: str, timeout: int = 10) -> str:
    """Execute Python code in subprocess. Returns output (max 200 chars)."""
    blocked = ["subprocess", "venv", "shutil", "os.system", "__import__", "open("]
    if any(b in code for b in blocked):
        return "Error: blocked operation"

    full_code = SAFE_IMPORTS + "\n" + code
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


def process_code_in_samples(gen_texts: list[str]) -> list[str]:
    """Execute code for samples that stopped at ```output."""
    results = []
    for text in gen_texts:
        if text.rstrip().endswith(("```output", "```output\n")):
            code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
            if code_blocks:
                output = execute_code(code_blocks[-1].strip())
                text = text.rstrip()
                if not text.endswith("\n"):
                    text += "\n"
                text += output + "\n```\n"
            else:
                text += "\nNo code found\n```\n"
        results.append(text)
    return results


def sc_tir_predict(llm, tokenizer, problem: str, system_prompt: str, user_template: str,
                   num_samples: int = 8, num_rounds: int = 4,
                   temperature: float = 0.8, max_tokens: int = 2048, seed: int = 42) -> int:
    """Batched SC-TIR prediction with majority voting."""
    from vllm import SamplingParams

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(problem=problem)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{system_prompt}\n\n{user_template.format(problem=problem)}"

    gen_texts = [prompt] * num_samples

    sp = SamplingParams(
        temperature=temperature, top_p=0.95,
        max_tokens=max_tokens, n=1,
        stop=["```output\n", "```output"],
        include_stop_str_in_output=True,
        seed=seed,
    )

    for round_idx in range(num_rounds):
        outputs = llm.generate(gen_texts, sampling_params=sp, use_tqdm=False)
        gen_texts = [gen_texts[i] + outputs[i].outputs[0].text for i in range(len(outputs))]

        needs_exec = sum(1 for t in gen_texts if t.rstrip().endswith(("```output", "```output\n")))
        if needs_exec == 0:
            break

        gen_texts = process_code_in_samples(gen_texts)

    answers = [extract_answer(text) for text in gen_texts]
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0
    return Counter(valid).most_common(1)[0][0]
