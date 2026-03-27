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
            logger.warning(
                "Trial %d: estimated runtime %.0fs exceeds budget %ds, pruning", trial.number, est, TIME_BUDGET
            )
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
    parser.add_argument(
        "--score-wait", type=int, default=120, help="Seconds to wait for score propagation (default: 2min)"
    )
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
