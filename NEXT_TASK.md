# Next Task: AIMO Prize 3

## Current Status (2026-03-25)

- **v18 Submit**: ERROR — Dataset がマウントされなかった（エディタで追加しても Submit 時に反映されない）
- **v17 通常実行**: SUCCESS（internet ON でテスト）
- **根本問題**: エディタで Dataset を追加しても Submit 時の実行環境に反映されない

## 最優先: Submit を成功させる

### 仮説: Save Version → Submit の順序が必要

エディタで Dataset 追加後、**先に Save Version** してから Submit する必要がある可能性。

### 手順（明日実行）

1. https://www.kaggle.com/code/koheimiki/aimo-3-baseline-deepseek-r1-majority-voting/edit を開く
2. Input に以下が含まれていることを確認:
   - COMPETITIONS: AI Mathematical Olympiad - Progress Prize 3
   - DATASETS: VLLM 0.13.0 with Dependencies for Offline Install
   - MODELS: DeepSeek R1 0528
3. Session options > Internet OFF 確認
4. **Save Version** をクリック（Quick Save ではなく Save Version）
5. Save 完了を待つ
6. **Submit to competition > Submit**

### 代替案: API push のバージョンから Submit

`kaggle kernels push` で作成したバージョン（dataset_sources 含む）を、エディタの Versions パネルから選択して Submit できるか確認。

### 代替案2: Kaggle API で直接 Submit

```bash
# 未確認 — Kaggle API に submit コマンドがあるか調査
kaggle competitions submit -c ai-mathematical-olympiad-progress-prize-3 -f submission.parquet -m "test"
```

## SC-TIR 実装済み

v17/v18 で Numina パターンの SC-TIR を実装:
- `stop=["```output\n"]` で vLLM バッチ停止
- subprocess でコード実行（timeout 10s）
- 8 サンプル × 4 ラウンド
- sympy/numpy 利用可能
- フォールバック: TIR 失敗時は直接生成

## Experiment PDCA

```bash
python scripts/evaluate_local.py --temperature 0.3 --num_samples 4
python scripts/evaluate_local.py --tir
python scripts/grid_search.py --max_problems 3
python scripts/show_experiments.py
```

## Key Lessons

1. **エディタの Input 追加だけでは Submit に反映されない** — Save Version が必要かもしれない
2. **API push の dataset_sources もエディタ Submit には反映されない**
3. **vllm はプリインストールされていない** → オフラインインストール用 Dataset 必須
4. **1日1回 Submit** — 慎重に

## Important Notes

- **Deadline: 2026-04-15** — 残り 21 日
- **Penalized accuracy**: seed=42 で対策済み
- **5時間 GPU 制限**: 110問 × 8サンプル × 4ラウンド
