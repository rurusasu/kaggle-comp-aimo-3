"""Submission module for AIMO3 Kaggle code competition.

On Kaggle, the submission uses the kaggle_evaluation API which serves
problems one at a time. This module provides both:
1. Local submission CSV creation (for testing)
2. The Kaggle API integration pattern
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import Config


def create_submission(
    cfg: Config,
    ids: Sequence,
    predictions: Sequence[int],
) -> Path:
    """Create submission CSV for local testing."""
    cfg.submissions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = cfg.submissions_dir / f"submission_{timestamp}.csv"
    df = pd.DataFrame({"id": ids, "answer": predictions})
    df.to_csv(path, index=False)
    return path
