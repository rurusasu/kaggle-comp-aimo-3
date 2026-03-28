from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    competition_name: str = "ai-mathematical-olympiad-progress-prize-3"
    seed: int = 42

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    logs_dir: Path = Path("logs")

    # LLM settings
    model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    model_dtype: str = "bfloat16"
    max_new_tokens: int = 16384
    max_model_len: int = 20480
    temperature: float = 0.6
    top_p: float = 0.95
    num_samples: int = 8
    gpu_memory_utilization: float = 0.92

    # Time budget (seconds) - GPU notebook has 5h = 18000s total
    time_budget_total: int = 17400  # leave 10min buffer
    problem_count: int = 110

    # Answer extraction
    answer_range_min: int = 0
    answer_range_max: int = 99999

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.logs_dir = Path(self.logs_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def submissions_dir(self) -> Path:
        return self.output_dir / "submissions"

    @property
    def oof_dir(self) -> Path:
        return self.output_dir / "oof"
