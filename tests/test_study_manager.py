import json
import tempfile
from pathlib import Path

import optuna
import optuna.storages

from src.optimizer.study_manager import create_or_load_study, save_trial_detail, show_best


def _dispose_storage(db_path: Path) -> None:
    """Dispose SQLite storage connections to allow temp dir cleanup on Windows."""
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
    storage.remove_session()


def test_create_new_study():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        study = create_or_load_study("test-study", db_path)
        assert study.study_name == "test-study"
        assert study.direction.name == "MAXIMIZE"
        assert len(study.trials) == 0
        _dispose_storage(db_path)


def test_load_existing_study():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
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
        _dispose_storage(db_path)


def test_save_trial_detail():
    with tempfile.TemporaryDirectory() as tmpdir:
        trials_dir = Path(tmpdir) / "trials"
        params = {"NUM_SAMPLES": 8, "TEMPERATURE": 0.5}
        save_trial_detail(trials_dir, trial_number=0, params=params, score=0.45, status="complete")

        detail_path = trials_dir / "trial_0000.json"
        assert detail_path.exists()

        data = json.loads(detail_path.read_text())
        assert data["trial_number"] == 0
        assert data["params"]["NUM_SAMPLES"] == 8
        assert data["score"] == 0.45
        assert data["status"] == "complete"


def test_show_best_no_trials(capsys):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        create_or_load_study("test-study", db_path)
        show_best(db_path, "test-study")
        captured = capsys.readouterr()
        assert "No completed trials" in captured.out
        _dispose_storage(db_path)
