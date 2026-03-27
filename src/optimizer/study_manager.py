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
