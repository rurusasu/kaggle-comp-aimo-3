# Next Task: AIMO Prize 3

## Status

- Kaggle Notebook v4 はコード完成（kaggle_evaluation API 対応、lazy loading、majority voting）
- DeepSeek-R1-0528-Qwen3-8B を Kaggle model source として設定済み
- v4 の通常実行は `run_local_gateway` モードで test.csv not found エラー（これは正常 — Submit 時のみ `serve()` が動く）
- **Web UI から Submit to Competition が必要**

## Step 1: Submit to Competition

**この手順はブラウザで実行する必要がある:**

1. https://www.kaggle.com/code/koheimiki/aimo-3-baseline-deepseek-r1-majority-voting/edit を開く
2. サイドバーの **Submit to competition** を展開
3. **Submit** ボタンをクリック
4. GPU として **GPU T4 x2** または **GPU P100** を選択（8B モデルは T4 x2 で動くはず）

### Submit の仕組み
- Submit 時に `KAGGLE_IS_COMPETITION_RERUN` 環境変数が設定される
- これにより `inference_server.serve()` が呼ばれ、API 経由で問題が配信される
- `run_local_gateway` は呼ばれない（test.csv not found エラーは発生しない）

## Step 2: 結果確認

```bash
uv run kaggle competitions submissions -c ai-mathematical-olympiad-progress-prize-3
```

## Step 3: 改善案（Submit 成功後）

### GPU メモリ問題が発生した場合
- `NUM_SAMPLES` を 16 → 8 に削減
- `MAX_NEW_TOKENS` を 32768 → 16384 に削減
- `GPU_MEMORY_UTILIZATION` を 0.90 → 0.95 に増加
- `dtype` を `bfloat16` → `float16` に変更

### スコア改善
1. **Prompt engineering**: reference.csv の 10 問を few-shot examples として使用
2. **Problem type detection**: 代数/組合せ/幾何/整数論ごとに異なるプロンプト
3. **Adaptive sampling**: 簡単な問題は N=4、難しい問題は N=32
4. **Answer extraction の改善**: `\boxed{}` 以外のパターン（"The answer is", "= " 等）
5. **Larger model**: Qwen3-32B や DeepSeek-R1-0528 の大きいバリアント（GPU メモリと時間の許す限り）

## Important Notes

- **Deadline: 2026-04-15** — 残り 23 日
- **Penalized accuracy**: 2 回実行され、両方で同じ答えを出す必要がある。temperature=0 なら確定的だが、majority voting は非決定的。温度を下げるか seed を固定。
- **5 時間 GPU 制限**: 110 問 × 16 サンプル。1 問あたり ~2.7 分が上限。
- **コードとデータの公開が賞金条件**。
- **docs/official-demo.md** に公式 Demo のコード全文あり。必ず参照すること。
