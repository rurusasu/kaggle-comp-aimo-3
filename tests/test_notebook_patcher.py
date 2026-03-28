import tempfile
from pathlib import Path

from src.optimizer.notebook_patcher import patch_notebook, read_current_params

SAMPLE_NOTEBOOK = """\
import os

# Configuration
NUM_SAMPLES = 8              # parallel candidates per problem
NUM_TIR_ROUNDS = 4           # max code execution rounds
MAX_NEW_TOKENS = 2048        # per generation step
TEMPERATURE = 0.8            # higher diversity
CODE_TIMEOUT = 10            # seconds per code execution

SYSTEM_PROMPT = "You are a mathematician."
"""


def _write_tmp(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def test_patch_notebook_replaces_all_params():
    path = _write_tmp(SAMPLE_NOTEBOOK)
    params = {
        "NUM_SAMPLES": 16,
        "NUM_TIR_ROUNDS": 2,
        "MAX_NEW_TOKENS": 4096,
        "TEMPERATURE": 0.3,
        "CODE_TIMEOUT": 20,
    }
    patch_notebook(path, params)
    content = path.read_text()

    assert "NUM_SAMPLES = 16" in content
    assert "NUM_TIR_ROUNDS = 2" in content
    assert "MAX_NEW_TOKENS = 4096" in content
    assert "TEMPERATURE = 0.3" in content
    assert "CODE_TIMEOUT = 20" in content


def test_patch_notebook_preserves_other_lines():
    path = _write_tmp(SAMPLE_NOTEBOOK)
    params = {"NUM_SAMPLES": 16, "NUM_TIR_ROUNDS": 2, "MAX_NEW_TOKENS": 4096, "TEMPERATURE": 0.3, "CODE_TIMEOUT": 20}
    patch_notebook(path, params)
    content = path.read_text()

    assert "SYSTEM_PROMPT" in content
    assert "import os" in content


def test_patch_notebook_validates_all_params_applied():
    path = _write_tmp(SAMPLE_NOTEBOOK)
    params = {"NUM_SAMPLES": 16}  # missing other params — should still work (partial patch)
    patch_notebook(path, params)
    content = path.read_text()

    assert "NUM_SAMPLES = 16" in content
    assert "NUM_TIR_ROUNDS = 4" in content  # unchanged


def test_read_current_params():
    path = _write_tmp(SAMPLE_NOTEBOOK)
    params = read_current_params(path)

    assert params["NUM_SAMPLES"] == 8
    assert params["NUM_TIR_ROUNDS"] == 4
    assert params["MAX_NEW_TOKENS"] == 2048
    assert params["TEMPERATURE"] == 0.8
    assert params["CODE_TIMEOUT"] == 10
