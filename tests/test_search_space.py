from unittest.mock import MagicMock

from src.optimizer.search_space import define_search_space, estimate_runtime


def test_define_search_space_returns_all_params():
    trial = MagicMock()
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    trial.suggest_float.return_value = 0.5

    params = define_search_space(trial)

    assert "NUM_SAMPLES" in params
    assert "NUM_TIR_ROUNDS" in params
    assert "MAX_NEW_TOKENS" in params
    assert "TEMPERATURE" in params
    assert "CODE_TIMEOUT" in params
    assert len(params) == 5


def test_define_search_space_values_from_trial():
    trial = MagicMock()
    trial.suggest_categorical.side_effect = lambda name, choices: {
        "NUM_SAMPLES": 16,
        "NUM_TIR_ROUNDS": 3,
        "MAX_NEW_TOKENS": 4096,
        "CODE_TIMEOUT": 15,
    }[name]
    trial.suggest_float.return_value = 0.3

    params = define_search_space(trial)

    assert params["NUM_SAMPLES"] == 16
    assert params["NUM_TIR_ROUNDS"] == 3
    assert params["MAX_NEW_TOKENS"] == 4096
    assert params["TEMPERATURE"] == 0.3
    assert params["CODE_TIMEOUT"] == 15


def test_estimate_runtime_within_budget():
    params = {"NUM_SAMPLES": 4, "NUM_TIR_ROUNDS": 1, "MAX_NEW_TOKENS": 1024, "TEMPERATURE": 0.5, "CODE_TIMEOUT": 5}
    estimated = estimate_runtime(params)
    assert estimated < 17400


def test_estimate_runtime_exceeds_budget():
    params = {"NUM_SAMPLES": 32, "NUM_TIR_ROUNDS": 5, "MAX_NEW_TOKENS": 4096, "TEMPERATURE": 0.5, "CODE_TIMEOUT": 20}
    estimated = estimate_runtime(params)
    assert estimated > 17400


def test_estimate_runtime_proportional_to_samples():
    base = {"NUM_TIR_ROUNDS": 4, "MAX_NEW_TOKENS": 2048, "TEMPERATURE": 0.5, "CODE_TIMEOUT": 10}
    t_low = estimate_runtime({**base, "NUM_SAMPLES": 4})
    t_high = estimate_runtime({**base, "NUM_SAMPLES": 32})
    assert t_high > t_low
