import pandas as pd

from src.config import Config


def load_reference(cfg: Config) -> pd.DataFrame:
    """Load reference problems with known answers (for local testing)."""
    path = cfg.raw_dir / "reference.csv"
    return pd.read_csv(path)


def load_test(cfg: Config) -> pd.DataFrame:
    """Load test problems."""
    path = cfg.raw_dir / "test.csv"
    return pd.read_csv(path)
