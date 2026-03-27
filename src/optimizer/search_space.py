"""Search space definition and runtime estimation for auto-optimize."""

import optuna

# 110 problems, base time per problem with 8 samples / 4 TIR rounds ≈ 90s on H100
_BASE_TIME_PER_PROBLEM = 90.0
_BASE_SAMPLES = 8
_BASE_TIR_ROUNDS = 4
_BASE_MAX_TOKENS = 2048
_PROBLEM_COUNT = 110
_MODEL_LOAD_TIME = 120.0  # ~2 min for model loading


def define_search_space(trial: optuna.Trial) -> dict:
    """Suggest parameters from the search space for a single trial."""
    return {
        "NUM_SAMPLES": trial.suggest_categorical("NUM_SAMPLES", [4, 8, 16, 32]),
        "NUM_TIR_ROUNDS": trial.suggest_categorical("NUM_TIR_ROUNDS", [1, 2, 3, 4, 5]),
        "MAX_NEW_TOKENS": trial.suggest_categorical("MAX_NEW_TOKENS", [1024, 2048, 3072, 4096]),
        "TEMPERATURE": trial.suggest_float("TEMPERATURE", 0.1, 1.0, step=0.1),
        "CODE_TIMEOUT": trial.suggest_categorical("CODE_TIMEOUT", [5, 10, 15, 20]),
    }


def estimate_runtime(params: dict) -> float:
    """Estimate total runtime in seconds for 110 problems with given params.

    Uses linear scaling from the baseline (8 samples, 4 TIR rounds, 2048 tokens).
    Returns estimated seconds.
    """
    sample_factor = params["NUM_SAMPLES"] / _BASE_SAMPLES
    tir_factor = params["NUM_TIR_ROUNDS"] / _BASE_TIR_ROUNDS
    token_factor = (params["MAX_NEW_TOKENS"] / _BASE_MAX_TOKENS) ** 0.5  # sub-linear scaling

    # CODE_TIMEOUT adds a small fixed cost per TIR round, not multiplicative
    code_overhead = params["NUM_TIR_ROUNDS"] * (params["CODE_TIMEOUT"] - 10) * params["NUM_SAMPLES"]

    per_problem = _BASE_TIME_PER_PROBLEM * sample_factor * tir_factor * token_factor
    return _MODEL_LOAD_TIME + _PROBLEM_COUNT * per_problem + code_overhead
