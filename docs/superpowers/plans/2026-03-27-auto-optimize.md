# Auto-Optimize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optuna + Kaggle CLI で推論パラメータを自動最適化する完全自動ループを構築する。

**Architecture:** Optuna (TPE sampler) が trial を生成し、notebook.py のパラメータ定数を正規表現で書き換え、`kaggle kernels push` で実行、ポーリングでスコアを取得して Optuna にフィードバックする。全 trial は SQLite に永続化される。

**Tech Stack:** Python 3.14, Optuna, kaggle CLI, subprocess, re, argparse

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/optimizer/__init__.py` | Package marker |
| Create | `src/optimizer/search_space.py` | 探索空間定義 + ランタイム推定 |
| Create | `src/optimizer/notebook_patcher.py` | notebook.py パラメータ書き換え |
| Create | `src/optimizer/kaggle_runner.py` | Kaggle CLI wrapper |
| Create | `src/optimizer/study_manager.py` | Optuna study 管理 |
| Create | `scripts/auto_optimize.py` | メインスクリプト |
| Create | `tests/test_search_space.py` | search_space テスト |
| Create | `tests/test_notebook_patcher.py` | notebook_patcher テスト |
| Create | `tests/test_kaggle_runner.py` | kaggle_runner テスト |
| Create | `tests/test_study_manager.py` | study_manager テスト |
| Modify | `pyproject.toml` | optuna 依存追加 |
| Modify | `Taskfile.yml` | optimize タスク追加 |

---

### Task 1: 依存関係の追加

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: pyproject.toml に optuna を追加**

```toml
[project.optional-dependencies]
dl = [
    "torch>=2.2",
    "torchvision>=0.17",
]
optimize = [
    "optuna>=4.0",
    "optuna-dashboard>=0.16",
]
dev = [
    "ruff>=0.4",
    "pytest>=8.0",
    "kaggle>=1.6",
    "optuna>=4.0",
]
```

- [ ] **Step 2: uv sync で依存をインストール**

Run: `uv sync --dev`
Expected: optuna がインストールされる

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add optuna dependency for auto-optimize"
```

---

### Task 2: search_space モジュール

**Files:**
- Create: `src/optimizer/__init__.py`
- Create: `src/optimizer/search_space.py`
- Create: `tests/test_search_space.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_search_space.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search_space.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create package and implement**

```python
# src/optimizer/__init__.py
```

```python
# src/optimizer/search_space.py
"""Search space definition and runtime estimation for auto-optimize."""

import optuna

# 110 problems, base time per problem with 8 samples / 4 TIR rounds ≈ 120s on H100
_BASE_TIME_PER_PROBLEM = 120.0
_BASE_SAMPLES = 8
_BASE_TIR_ROUNDS = 4
_BASE_MAX_TOKENS = 2048
_BASE_CODE_TIMEOUT = 10
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
    token_factor = params["MAX_NEW_TOKENS"] / _BASE_MAX_TOKENS
    timeout_factor = params["CODE_TIMEOUT"] / _BASE_CODE_TIMEOUT

    per_problem = _BASE_TIME_PER_PROBLEM * sample_factor * tir_factor * token_factor * timeout_factor
    return _MODEL_LOAD_TIME + _PROBLEM_COUNT * per_problem
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search_space.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/__init__.py src/optimizer/search_space.py tests/test_search_space.py
git commit -m "feat: add search_space module with parameter definitions and runtime estimation"
```

---

### Task 3: notebook_patcher モジュール

**Files:**
- Create: `src/optimizer/notebook_patcher.py`
- Create: `tests/test_notebook_patcher.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_notebook_patcher.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_notebook_patcher.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

```python
# src/optimizer/notebook_patcher.py
"""Patch notebook.py parameter constants via regex replacement."""

import re
from pathlib import Path

# Parameters to patch: name -> type converter
_PARAM_TYPES: dict[str, type] = {
    "NUM_SAMPLES": int,
    "NUM_TIR_ROUNDS": int,
    "MAX_NEW_TOKENS": int,
    "TEMPERATURE": float,
    "CODE_TIMEOUT": int,
}


def patch_notebook(notebook_path: Path, params: dict) -> None:
    """Replace parameter constants in notebook.py with new values.

    Matches lines like `NUM_SAMPLES = 8  # comment` and replaces the value
    while preserving the inline comment.
    """
    content = notebook_path.read_text(encoding="utf-8")

    for name, value in params.items():
        if name not in _PARAM_TYPES:
            raise ValueError(f"Unknown parameter: {name}")
        pattern = rf"^({name}\s*=\s*)(\S+)(\s*#.*)?$"
        formatted = str(value) if isinstance(value, int) else f"{value}"
        replacement = rf"\g<1>{formatted}\g<3>"
        content, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
        if count == 0:
            raise ValueError(f"Parameter {name} not found in {notebook_path}")

    notebook_path.write_text(content, encoding="utf-8")


def read_current_params(notebook_path: Path) -> dict:
    """Read current parameter values from notebook.py."""
    content = notebook_path.read_text(encoding="utf-8")
    result = {}
    for name, typ in _PARAM_TYPES.items():
        match = re.search(rf"^{name}\s*=\s*(\S+)", content, re.MULTILINE)
        if match:
            result[name] = typ(match.group(1))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_notebook_patcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/notebook_patcher.py tests/test_notebook_patcher.py
git commit -m "feat: add notebook_patcher module for regex-based parameter replacement"
```

---

### Task 4: kaggle_runner モジュール

**Files:**
- Create: `src/optimizer/kaggle_runner.py`
- Create: `tests/test_kaggle_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kaggle_runner.py
import json
from unittest.mock import MagicMock, patch

from src.optimizer.kaggle_runner import KaggleRunner


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_push_returns_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="Kernel version 5 pushed.", stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    result = runner.push("kaggle-notebook/")
    assert result["success"] is True
    mock_run.assert_called_once()


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_push_failure_raises(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="409 Conflict")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    result = runner.push("kaggle-notebook/")
    assert result["success"] is False


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_get_status_complete(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout='{"status": "complete"}', stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    status = runner.get_status()
    assert status == "complete"


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_get_status_running(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout='{"status": "running"}', stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    status = runner.get_status()
    assert status == "running"


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_latest_score(mock_run):
    submissions_output = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.csv,2026-03-27,auto,complete,0.45,0.0\n"
    )
    mock_run.return_value = MagicMock(returncode=0, stdout=submissions_output, stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    score = runner.fetch_latest_score()
    assert score == 0.45


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_latest_score_no_submissions(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="fileName,date,description,status,publicScore,privateScore\n", stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    score = runner.fetch_latest_score()
    assert score is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_kaggle_runner.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

```python
# src/optimizer/kaggle_runner.py
"""Kaggle CLI wrapper for push, status polling, and score retrieval."""

import csv
import io
import json
import logging
import subprocess
import time

logger = logging.getLogger(__name__)


class KaggleRunner:
    def __init__(self, kernel_id: str, competition: str):
        self.kernel_id = kernel_id
        self.competition = competition

    def push(self, notebook_dir: str, retries: int = 3) -> dict:
        """Push notebook to Kaggle. Returns {"success": bool, "stdout": str, "stderr": str}."""
        for attempt in range(retries):
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", notebook_dir],
                capture_output=True,
                text=True,
                env=_utf8_env(),
            )
            if result.returncode == 0:
                logger.info("Push succeeded: %s", result.stdout.strip())
                return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
            logger.warning("Push attempt %d failed: %s", attempt + 1, result.stderr.strip())
            if attempt < retries - 1:
                time.sleep(10)
        return {"success": False, "stdout": result.stdout, "stderr": result.stderr}

    def get_status(self) -> str:
        """Get kernel execution status. Returns one of: complete, running, error, cancelled, queued."""
        result = subprocess.run(
            ["kaggle", "kernels", "status", self.kernel_id],
            capture_output=True,
            text=True,
            env=_utf8_env(),
        )
        if result.returncode != 0:
            logger.error("Status check failed: %s", result.stderr)
            return "error"
        try:
            data = json.loads(result.stdout)
            return data.get("status", "unknown")
        except json.JSONDecodeError:
            # Fallback: parse text output like "has status \"complete\""
            stdout = result.stdout.lower()
            for s in ("complete", "running", "error", "cancelled", "queued"):
                if s in stdout:
                    return s
            return "unknown"

    def poll_until_complete(self, timeout: int = 21600, interval: int = 300) -> str:
        """Poll kernel status until terminal state or timeout.

        Returns final status string.
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status()
            logger.info("Kernel status: %s (%.0fs elapsed)", status, time.time() - start)
            if status in ("complete", "error", "cancelled"):
                return status
            time.sleep(interval)
        logger.warning("Polling timed out after %ds", timeout)
        return "timeout"

    def fetch_latest_score(self, wait: int = 120) -> float | None:
        """Fetch the latest public score from competition submissions.

        Waits `wait` seconds for score to propagate, then parses CSV output.
        Returns score as float, or None if not available.
        """
        if wait > 0:
            logger.info("Waiting %ds for score to propagate...", wait)
            time.sleep(wait)

        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", self.competition, "--csv"],
            capture_output=True,
            text=True,
            env=_utf8_env(),
        )
        if result.returncode != 0:
            logger.error("Failed to fetch submissions: %s", result.stderr)
            return None

        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        if not rows:
            return None

        latest = rows[0]  # most recent submission
        try:
            return float(latest["publicScore"])
        except (KeyError, ValueError):
            logger.warning("Could not parse score from: %s", latest)
            return None


def _utf8_env():
    """Return env dict with PYTHONUTF8=1 for Windows compatibility."""
    import os

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_kaggle_runner.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/kaggle_runner.py tests/test_kaggle_runner.py
git commit -m "feat: add kaggle_runner module for push, polling, and score retrieval"
```

---

### Task 5: study_manager モジュール

**Files:**
- Create: `src/optimizer/study_manager.py`
- Create: `tests/test_study_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_study_manager.py
import tempfile
from pathlib import Path

from src.optimizer.study_manager import create_or_load_study, show_best, save_trial_detail


def test_create_new_study():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        study = create_or_load_study("test-study", db_path)
        assert study.study_name == "test-study"
        assert study.direction.name == "MAXIMIZE"
        assert len(study.trials) == 0


def test_load_existing_study():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        study1 = create_or_load_study("test-study", db_path)
        study1.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                values=[0.5],
                params={"TEMPERATURE": 0.5},
                distributions={"TEMPERATURE": optuna.distributions.FloatDistribution(0.1, 1.0)},
            )
        )
        study2 = create_or_load_study("test-study", db_path)
        assert len(study2.trials) == 1


def test_save_trial_detail():
    with tempfile.TemporaryDirectory() as tmpdir:
        trials_dir = Path(tmpdir) / "trials"
        params = {"NUM_SAMPLES": 8, "TEMPERATURE": 0.5}
        save_trial_detail(trials_dir, trial_number=0, params=params, score=0.45, status="complete")

        detail_path = trials_dir / "trial_0000.json"
        assert detail_path.exists()

        import json
        data = json.loads(detail_path.read_text())
        assert data["trial_number"] == 0
        assert data["params"]["NUM_SAMPLES"] == 8
        assert data["score"] == 0.45
        assert data["status"] == "complete"


def test_show_best_no_trials(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        create_or_load_study("test-study", db_path)
        show_best(db_path, "test-study")
        captured = capsys.readouterr()
        assert "No completed trials" in captured.out
```

Add import at top of test file:

```python
import optuna
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_study_manager.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

```python
# src/optimizer/study_manager.py
"""Optuna study management: create, load, show results."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import optuna

logger = logging.getLogger(__name__)


def create_or_load_study(study_name: str, db_path: Path) -> optuna.Study:
    """Create a new study or load existing one from SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )


def save_trial_detail(trials_dir: Path, trial_number: int, params: dict, score: float | None, status: str) -> None:
    """Save detailed trial information as JSON."""
    trials_dir.mkdir(parents=True, exist_ok=True)
    detail = {
        "trial_number": trial_number,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "params": params,
        "score": score,
        "status": status,
    }
    path = trials_dir / f"trial_{trial_number:04d}.json"
    path.write_text(json.dumps(detail, indent=2))
    logger.info("Trial %d detail saved to %s", trial_number, path)


def show_best(db_path: Path | None = None, study_name: str = "auto-optimize") -> None:
    """Print the best trial from a study."""
    if db_path is None:
        db_path = Path("logs/auto_optimize/optuna_study.db")
    if not db_path.exists():
        print(f"No study database found at {db_path}")
        return

    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f"Study '{study_name}' not found in {db_path}")
        return

    if len(study.trials) == 0:
        print("No completed trials yet.")
        return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials yet.")
        return

    best = study.best_trial
    print(f"Best trial: #{best.number}")
    print(f"  Score: {best.value}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"  Total trials: {len(study.trials)} ({len(completed)} completed)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_study_manager.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/study_manager.py tests/test_study_manager.py
git commit -m "feat: add study_manager module for Optuna study lifecycle"
```

---

### Task 6: auto_optimize メインスクリプト

**Files:**
- Create: `scripts/auto_optimize.py`

- [ ] **Step 1: Implement the main script**

```python
# scripts/auto_optimize.py
"""Automated parameter optimization loop using Optuna + Kaggle CLI.

Usage:
    python scripts/auto_optimize.py --study-name exp-v1
    python scripts/auto_optimize.py --study-name exp-v1 --n-trials 20
    python scripts/auto_optimize.py --study-name exp-v1 --resume
"""

import argparse
import logging
import sys
from pathlib import Path

import optuna

from src.optimizer.kaggle_runner import KaggleRunner
from src.optimizer.notebook_patcher import patch_notebook, read_current_params
from src.optimizer.search_space import define_search_space, estimate_runtime
from src.optimizer.study_manager import create_or_load_study, save_trial_detail, show_best

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOK_PATH = Path("kaggle-notebook/notebook.py")
NOTEBOOK_DIR = "kaggle-notebook/"
KERNEL_ID = "koheimiki/aimo-3-baseline-deepseek-r1-majority-voting"
COMPETITION = "ai-mathematical-olympiad-progress-prize-3"
DB_DIR = Path("logs/auto_optimize")
TRIALS_DIR = DB_DIR / "trials"
TIME_BUDGET = 17400  # 5h - 10min buffer


def make_objective(runner: KaggleRunner, poll_timeout: int, poll_interval: int, score_wait: int):
    """Create the Optuna objective function with the given runner config."""

    def objective(trial: optuna.Trial) -> float:
        # 1. Suggest parameters
        params = define_search_space(trial)
        logger.info("Trial %d: %s", trial.number, params)

        # 2. Check estimated runtime
        est = estimate_runtime(params)
        if est > TIME_BUDGET:
            logger.warning("Trial %d: estimated runtime %.0fs exceeds budget %ds, pruning", trial.number, est, TIME_BUDGET)
            raise optuna.TrialPruned(f"Estimated runtime {est:.0f}s exceeds {TIME_BUDGET}s")

        # 3. Patch notebook
        original_params = read_current_params(NOTEBOOK_PATH)
        patch_notebook(NOTEBOOK_PATH, params)
        logger.info("Trial %d: notebook patched", trial.number)

        try:
            # 4. Push to Kaggle
            push_result = runner.push(NOTEBOOK_DIR)
            if not push_result["success"]:
                logger.error("Trial %d: push failed: %s", trial.number, push_result["stderr"])
                raise optuna.TrialPruned(f"Push failed: {push_result['stderr']}")

            # 5. Poll until complete
            status = runner.poll_until_complete(timeout=poll_timeout, interval=poll_interval)
            if status != "complete":
                logger.warning("Trial %d: kernel status=%s, pruning", trial.number, status)
                save_trial_detail(TRIALS_DIR, trial.number, params, score=None, status=status)
                raise optuna.TrialPruned(f"Kernel status: {status}")

            # 6. Fetch score
            score = runner.fetch_latest_score(wait=score_wait)
            if score is None:
                logger.error("Trial %d: could not fetch score", trial.number)
                save_trial_detail(TRIALS_DIR, trial.number, params, score=None, status="score_missing")
                raise optuna.TrialPruned("Could not fetch score")

            logger.info("Trial %d: score=%.4f", trial.number, score)
            save_trial_detail(TRIALS_DIR, trial.number, params, score=score, status="complete")
            return score

        except optuna.TrialPruned:
            raise
        except Exception:
            # Restore original params on unexpected error
            patch_notebook(NOTEBOOK_PATH, original_params)
            raise

    return objective


def main():
    parser = argparse.ArgumentParser(description="Auto-optimize notebook parameters via Optuna + Kaggle CLI")
    parser.add_argument("--study-name", default="auto-optimize", help="Optuna study name")
    parser.add_argument("--n-trials", type=int, default=None, help="Max number of trials (default: unlimited)")
    parser.add_argument("--resume", action="store_true", help="Resume existing study (default: create new)")
    parser.add_argument("--poll-timeout", type=int, default=21600, help="Max seconds to wait for kernel (default: 6h)")
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds between status checks (default: 5min)")
    parser.add_argument("--score-wait", type=int, default=120, help="Seconds to wait for score propagation (default: 2min)")
    args = parser.parse_args()

    db_path = DB_DIR / "optuna_study.db"
    study = create_or_load_study(args.study_name, db_path)

    if not args.resume and len(study.trials) > 0:
        logger.error("Study '%s' already has %d trials. Use --resume to continue.", args.study_name, len(study.trials))
        sys.exit(1)

    runner = KaggleRunner(kernel_id=KERNEL_ID, competition=COMPETITION)
    objective = make_objective(runner, args.poll_timeout, args.poll_interval, args.score_wait)

    logger.info("Starting optimization: study=%s, n_trials=%s", args.study_name, args.n_trials or "unlimited")
    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    show_best(db_path, args.study_name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is importable**

Run: `uv run python -c "import scripts.auto_optimize; print('OK')"`

If this fails due to script not being a package, verify with:

Run: `uv run python scripts/auto_optimize.py --help`
Expected: Help text with --study-name, --n-trials, --resume options

- [ ] **Step 3: Commit**

```bash
git add scripts/auto_optimize.py
git commit -m "feat: add auto_optimize main script with Optuna + Kaggle CLI loop"
```

---

### Task 7: Taskfile.yml 更新

**Files:**
- Modify: `Taskfile.yml`

- [ ] **Step 1: Add optimize tasks to Taskfile.yml**

Append after the `clean` task:

```yaml
  optimize:
    desc: Run automated parameter optimization loop
    cmds:
      - uv run python scripts/auto_optimize.py {{.CLI_ARGS}}

  optimize-status:
    desc: Show optimization progress and best parameters
    cmds:
      - uv run python -c "from src.optimizer.study_manager import show_best; show_best()"

  optimize-dashboard:
    desc: Launch Optuna dashboard for visualization
    cmds:
      - uv run optuna-dashboard sqlite:///logs/auto_optimize/optuna_study.db
```

- [ ] **Step 2: Verify tasks are registered**

Run: `task --list`
Expected: optimize, optimize-status, optimize-dashboard が表示される

- [ ] **Step 3: Commit**

```bash
git add Taskfile.yml
git commit -m "feat: add optimize tasks to Taskfile.yml"
```

---

### Task 8: ログディレクトリと .gitignore

**Files:**
- Create: `logs/auto_optimize/.gitkeep`
- Create: `logs/auto_optimize/trials/.gitkeep`
- Modify: `.gitignore` (if needed)

- [ ] **Step 1: Create log directories with .gitkeep**

```bash
mkdir -p logs/auto_optimize/trials
touch logs/auto_optimize/.gitkeep
touch logs/auto_optimize/trials/.gitkeep
```

- [ ] **Step 2: Add SQLite and trial JSONs to .gitignore**

Check if `.gitignore` already covers these. If not, add:

```
logs/auto_optimize/optuna_study.db
logs/auto_optimize/trials/*.json
```

- [ ] **Step 3: Commit**

```bash
git add logs/auto_optimize/.gitkeep logs/auto_optimize/trials/.gitkeep .gitignore
git commit -m "chore: add auto_optimize log directories and gitignore rules"
```

---

### Task 9: 全体の結合テスト

**Files:**
- (no new files)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS (既存テスト + 新規テスト全て)

- [ ] **Step 2: Run linter**

Run: `task lint`
Expected: No errors

- [ ] **Step 3: Dry-run help check**

Run: `uv run python scripts/auto_optimize.py --help`
Expected:

```
usage: auto_optimize.py [-h] [--study-name STUDY_NAME] [--n-trials N_TRIALS] [--resume]
                        [--poll-timeout POLL_TIMEOUT] [--poll-interval POLL_INTERVAL]
                        [--score-wait SCORE_WAIT]
```

- [ ] **Step 4: Verify optimize-status task**

Run: `task optimize-status`
Expected: "No study database found" (まだ実行していないため)

- [ ] **Step 5: Commit (if any lint fixes)**

```bash
git add -A
git commit -m "fix: lint fixes for auto-optimize modules"
```
