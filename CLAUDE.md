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

## Current Approach

### Baseline Method

- **Model:** DeepSeek-R1-Qwen3-8B (open-source math reasoning model)
- **Inference:** vLLM for fast batched generation on GPU
- **Strategy:** Generate 16 chain-of-thought solutions per problem, extract integer answers via `\boxed{N}` pattern, select final answer by majority voting
- **Time management:** 5hr GPU limit with per-problem budget (~25 min max) and early exit when time is low

### Workflow

1. Download/prepare model weights (e.g., DeepSeek-R1-Qwen3-8B)
2. Upload model weights as a Kaggle Dataset
3. Push notebook via `kaggle kernels push kaggle-notebook/`
4. Notebook runs on Kaggle GPU (T4x2 or H100), no internet access
5. `kaggle_evaluation` API serves problems one-by-one; notebook returns integer answers

### File Layout

- `kaggle-notebook/` — Kaggle submission (kernel-metadata.json + notebook.py)
- `notebooks/` — Development notebooks (baseline_submission.py)
- `src/` — Reusable modules (config, model, dataset, evaluate, submit)
- `data/raw/` — Competition data (reference.csv, test.csv, sample_submission.csv)
- `docs/` — Competition documentation

### Training Data

- `reference.csv`: 10 example problems with integer answers (columns: id, problem, answer)
- `test.csv`: Placeholder test problems (columns: id, problem)
- Full evaluation runs 110 original math olympiad problems (algebra, combinatorics, geometry, number theory)

### How to Submit

```bash
# Push notebook to Kaggle
kaggle kernels push kaggle-notebook/
# Monitor status
kaggle kernels status koheimiki/aimo-3-baseline
```

### Improvement Ideas

- Stronger/larger models (DeepSeek-R1 distilled variants, Qwen3-32B if memory allows)
- Prompt engineering: few-shot examples from reference problems, problem-type-specific prompts
- Better answer extraction: handle more answer formats, confidence weighting
- Adaptive sampling: increase N for hard problems, decrease for easy ones
- Ensemble multiple models or prompting strategies
- Fine-tuning on math competition datasets (AIME, AMC, IMO Shortlist)

## Lessons Learned

### Kaggle Notebook 環境

1. **vllm がプリインストールされていない**: Notebook 冒頭で `pip install vllm` が必須。
2. **Internet 設定**: pip install のためには `enable_internet: true` が必要。ただし提出時はインターネット無効が求められるため、モデル weights は Kaggle Dataset として事前アップロードが必要。学習/デバッグ時のみ internet=true にする。
3. **kernel-metadata.json の id とタイトルの不一致**: Kaggle がタイトルから slug を自動生成し、push 時の id と一致しないと 409 Conflict になる。id は初回 push 後に生成された slug に合わせる。

### モデル選択

4. **モデル weights のアップロードが最大のボトルネック**: 8B モデルでも ~16GB。Kaggle Dataset のアップロードに時間がかかる。
5. **GPU 5時間制限**: H100 なら 8B モデル × 110 問 × 16 サンプル ≈ 3-4 時間で収まる見込み。T4 では厳しい。
6. **コードとデータセットの公開が必須**（賞金条件）: プライベートモデルは使えない。Hugging Face の公開モデルが前提。

### 評価

7. **Penalized accuracy**: 2 回実行され、両方で同じ答えを出す必要がある。非決定的な推論（temperature > 0）は不利。`temperature=0.0` を基本とし、majority voting で安定性を確保。

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
