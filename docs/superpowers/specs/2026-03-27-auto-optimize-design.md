# Auto-Optimize: Optuna + Kaggle CLI 自動実験ループ

## 概要

Optuna のベイズ最適化で推論パラメータを提案し、Kaggle Notebook にpush、スコアを自動取得してフィードバックする完全自動ループ。

## 背景・動機

AIMO-3 コンペ（数学オリンピック110問、締切 2026-04-15）で、推論パラメータの最適化を手動で回すのは非効率。Karpathy の autoresearch (2026年3月) や AIDE (WecoAI) のような自動実験ループのアプローチを、Kaggle Notebook 環境に適用する。

## 探索空間

| パラメータ | 現在値 | 探索範囲 | 型 |
|---|---|---|---|
| `NUM_SAMPLES` | 8 | 4, 8, 16, 32 | categorical |
| `NUM_TIR_ROUNDS` | 4 | 1, 2, 3, 4, 5 | categorical |
| `MAX_NEW_TOKENS` | 2048 | 1024, 2048, 3072, 4096 | categorical |
| `TEMPERATURE` | 0.8 | 0.1 〜 1.0 (step 0.1) | float |
| `CODE_TIMEOUT` | 10 | 5, 10, 15, 20 | categorical |

制約: `NUM_SAMPLES × NUM_TIR_ROUNDS` が大きすぎると5時間のGPU制限を超えるため、事前推定でフィルタする。

## アーキテクチャ

```
scripts/auto_optimize.py (メインループ)
  │
  ├─ src/optimizer/search_space.py     # 探索空間定義
  ├─ src/optimizer/notebook_patcher.py # notebook.py パラメータ書き換え
  ├─ src/optimizer/kaggle_runner.py    # push → poll → score取得
  └─ src/optimizer/study_manager.py    # Optuna study 管理
```

### ループフロー

```
Optuna study.optimize()
  → trial: パラメータ提案
  → estimate_runtime(): 5時間超 → prune
  → patch_notebook(): 正規表現でnotebook.pyの定数を書き換え
  → kaggle_push(): kaggle kernels push kaggle-notebook/
  → poll_until_complete(): 5分間隔ポーリング, 最大6時間
  → fetch_latest_score(): submissions一覧からスコア取得
  → save_trial_detail(): JSON保存
  → return score (maximize方向)
```

## ファイル構成

```
scripts/
  auto_optimize.py          # メインスクリプト (CLI entrypoint)

src/
  optimizer/
    __init__.py
    search_space.py         # パラメータ名, 範囲, 型の定義
    notebook_patcher.py     # notebook.py のパラメータ定数を正規表現で書き換え
    kaggle_runner.py        # kaggle CLI wrapper (push, status, submissions)
    study_manager.py        # Optuna study 作成/再開/ベスト表示

logs/
  auto_optimize/
    optuna_study.db         # Optuna SQLite (trial履歴永続化)
    trials/                 # trial毎の詳細JSON
```

## 各モジュールの責務

### search_space.py

- 探索空間を宣言的に定義
- `define_search_space(trial) -> dict` でパラメータセットを返す
- 実行時間の事前推定ロジック (`estimate_runtime`)

### notebook_patcher.py

- notebook.py の先頭にあるパラメータ定数 (`NUM_SAMPLES = 8` 等) を正規表現で書き換え
- 書き換え前のバックアップ不要（git管理のため）
- パッチ適用後のバリデーション（定数が正しく変わったか確認）

### kaggle_runner.py

- `kaggle_push()` — `kaggle kernels push kaggle-notebook/` を実行、バージョン番号を返す
- `poll_until_complete(timeout, interval)` — `kaggle kernels status` を定期実行、complete/error/cancelled を返す
- `fetch_latest_score(wait)` — `kaggle competitions submissions` からスコアをパース。submission反映待ちのために追加待機

### study_manager.py

- `create_or_load_study(name, db_path)` — SQLite永続化で study を作成または再開
- `show_best()` — ベストtrial のパラメータとスコアを表示
- maximize 方向の設定

### auto_optimize.py

- argparse CLI: `--study-name`, `--n-trials`, `--resume`
- objective 関数で全モジュールを連携
- KeyboardInterrupt で安全に停止（途中結果はSQLiteに保存済み）

## エラーリカバリ

| 障害 | 対応 |
|---|---|
| push 失敗 (409等) | 3回リトライ、失敗→trial prune |
| kernel error/cancelled | trial prune、次のtrialへ |
| スコア取得失敗 | submissions一覧から該当バージョンを特定して再取得 |
| スクリプトクラッシュ | SQLiteに途中保存済み、`--resume` で再開 |
| GPU週間制限到達 | ポーリングタイムアウト → 翌週再開 |

## CLI インターフェース

```bash
# 新規開始
task optimize -- --study-name exp-v1

# trial数を指定
task optimize -- --study-name exp-v1 --n-trials 20

# 中断後の再開
task optimize -- --study-name exp-v1 --resume

# ベスト結果表示
task optimize-status

# Optuna Dashboard
task optimize-dashboard
```

## Taskfile.yml 追加タスク

```yaml
optimize:
  desc: Run automated parameter optimization loop
  cmd: uv run python scripts/auto_optimize.py {{.CLI_ARGS}}

optimize-status:
  desc: Show optimization progress
  cmd: uv run python -c "from src.optimizer.study_manager import show_best; show_best()"

optimize-dashboard:
  desc: Launch Optuna dashboard
  cmd: uv run optuna-dashboard sqlite:///logs/auto_optimize/optuna_study.db
```

## 依存関係

- `optuna` — ベイズ最適化 (TPE sampler)
- `optuna-dashboard` — 可視化 (オプション)

## 将来の拡張

1. **プロンプト最適化**: search_space.py にプロンプトテンプレートのcategorical変数を追加
2. **LLM分析**: Optuna callback でN trial毎にLLMが探索空間を再提案
3. **コード変更**: notebook_patcher をautoresearch的なLLMコード書き換えに拡張

## 参考

- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) — 自動実験ループの先駆的実装
- [AIDE (WecoAI)](https://github.com/WecoAI/aideml) — Kaggle特化ツリーサーチエージェント
- [AI Scientist v2 (Sakana AI)](https://github.com/SakanaAI/AI-Scientist-v2) — 完全自動科学実験ループ
- [DSPy](https://dspy.ai/) — プロンプト最適化フレームワーク
- [Optuna](https://optuna.org/) — ベイズ最適化フレームワーク
