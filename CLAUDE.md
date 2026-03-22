# AI Mathematical Olympiad - Progress Prize 3

## Competition Info

- **URL:** https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3
- **Deadline:** 2026-04-15 23:59 UTC
- **Prize:** $2,207,152
- **Category:** Featured
- **Type:** Code Competition

## Task

数学オリンピックレベル（国内〜IMO水準）の110問のオリジナル数学問題をAIモデルで解く。分野: 代数、組合せ論、幾何学、整数論。

## Evaluation

- **Metric:** Penalized accuracy scores
- pass@1 ベースの精度評価

## Submission Format

- Kaggle Notebook での提出（コードコンペ）
- Submission Demo: https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo

## Special Rules

- コードとデータセットの公開が必須（賞金獲得の条件）
- GPU: 最大128基の H100 GPU が利用可能（Fields Model Initiative 提供）
- ファインチューニング用に提出前に利用可能

## Documentation

**IMPORTANT: Before starting any implementation work, you MUST read the relevant docs first.**

- [docs/overview.md](docs/overview.md) — Competition description, goal, background
- [docs/evaluation.md](docs/evaluation.md) — Evaluation metric, scoring methodology
- [docs/submission.md](docs/submission.md) — Submission format, file structure, requirements
- [docs/timeline.md](docs/timeline.md) — Important dates and deadlines
- [docs/rules.md](docs/rules.md) — Full competition rules
- [docs/prizes.md](docs/prizes.md) — Prize structure

### Required Reading Order

1. Before EDA or feature engineering → read `overview.md` and `evaluation.md`
2. Before building submission pipeline → read `submission.md`
3. Before using external data or models → read `rules.md`
4. Before final submission → read `timeline.md` to confirm deadlines

---

# Kaggle Competition Workspace

## Structure

- `src/config.py` — All configuration (paths, params, seed). Change settings HERE, not in other modules.
- `src/dataset.py` — Stateless data I/O. `load_train()` / `load_test()` return raw DataFrames.
- `src/features.py` — Feature engineering. Stateful transforms (fit on train only).
- `src/model.py` — Model train/predict/save/load.
- `src/evaluate.py` — CV splitter, metrics, experiment logging. Owns all writes to `logs/`.
- `src/submit.py` — Generates timestamped submission CSVs.
- `src/utils.py` — `set_seed()`, `Timer`.
- `scripts/train.py` — Training entrypoint. Runs full CV pipeline.
- `scripts/predict.py` — Inference entrypoint. Loads saved models, generates submission.

## Conventions

- Format with ruff (line-length=120, Python 3.14)
- Type hints encouraged
- Config changes go in `src/config.py` only
- Experiment logs go in `logs/` via `src/evaluate.py` only

## Commands

- `task setup` — Install deps + download data
- `task train` — Train models
- `task predict` — Generate predictions
- `task submit` — Submit to Kaggle
- `task lint` — Check code style
- `task test` — Run tests
