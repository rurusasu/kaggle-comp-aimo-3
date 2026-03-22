"""Math problem solver using vLLM with majority voting.

This module handles:
1. Loading an open-source reasoning LLM via vLLM
2. Constructing prompts for math olympiad problems
3. Generating multiple solutions per problem
4. Extracting integer answers and performing majority voting
"""

from __future__ import annotations

import re
from collections import Counter

from src.config import Config

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


def build_prompt(problem: str) -> str:
    """Build the user prompt for a math problem."""
    return USER_PROMPT_TEMPLATE.format(problem=problem)


def build_messages(problem: str) -> list[dict[str, str]]:
    """Build chat messages for a math problem."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(problem)},
    ]


def extract_answer(text: str, cfg: Config) -> int | None:
    """Extract integer answer from model output.

    Looks for \\boxed{N} pattern, falls back to last integer in text.
    Returns None if no valid answer found.
    """
    # Try \\boxed{...} pattern (last occurrence wins)
    boxed_matches = re.findall(r"\\boxed\{(\d+)\}", text)
    if boxed_matches:
        val = int(boxed_matches[-1])
        if cfg.answer_range_min <= val <= cfg.answer_range_max:
            return val

    # Fallback: last standalone integer in the text
    int_matches = re.findall(r"\b(\d+)\b", text)
    if int_matches:
        val = int(int_matches[-1])
        if cfg.answer_range_min <= val <= cfg.answer_range_max:
            return val

    return None


def majority_vote(answers: list[int | None]) -> int:
    """Select answer by majority vote. Falls back to 0 if no valid answers."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


def load_vllm_model(cfg: Config):
    """Load model using vLLM for fast inference.

    This should be called once at the start of inference.
    Returns (model, tokenizer) for use with vLLM.
    """
    from vllm import LLM, SamplingParams  # noqa: F401

    llm = LLM(
        model=cfg.model_name,
        dtype=cfg.model_dtype,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_new_tokens + 4096,  # input + output
        trust_remote_code=True,
    )
    return llm


def solve_problem(llm, problem: str, cfg: Config) -> int:
    """Solve a single math problem using majority voting over multiple samples.

    Args:
        llm: vLLM model instance
        problem: LaTeX math problem text
        cfg: Configuration

    Returns:
        Integer answer (0-99999)
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        n=cfg.num_samples,
    )

    messages = build_messages(problem)

    # Use chat template if available
    try:
        outputs = llm.chat([messages], sampling_params=sampling_params)
    except Exception:
        # Fallback to plain text prompt
        prompt = f"{SYSTEM_PROMPT}\n\n{build_prompt(problem)}"
        outputs = llm.generate([prompt], sampling_params=sampling_params)

    answers = []
    for output in outputs:
        for completion in output.outputs:
            answer = extract_answer(completion.text, cfg)
            answers.append(answer)

    return majority_vote(answers)
