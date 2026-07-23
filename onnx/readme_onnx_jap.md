# onnx README (Japanese)

ONNX のエクスポート・検証・解析スクリプトまとめです。  
学習・DataLoader は [`../fine-tuning/`](../fine-tuning/) を参照してください。

解析結果 JSON の保存先はリポジトリ直下の `onnx_analysis/` です。

---

## export_onnx.py

TensorFlow Checkpoint → 単一 ONNX。

```bash
python onnx/export_onnx.py

python onnx/export_onnx.py --checkpoint-dir fine_tuned_model/M-LSD_512_large_ft_50_000125

python onnx/export_onnx.py \
  --checkpoint-dir fine_tuned_model/M-LSD_512_tiny_ft_100_00005 \
  --backbone-type MLSD

python onnx/export_onnx.py \
  --checkpoint-dir ckpt_models/M-LSD_320_tiny \
  --input-size 320 \
  --backbone-type MLSD
```

| 引数 | 説明 | default |
|---|---|---|
| `--checkpoint-dir` | チェックポイント（親を渡すと最新を選択） | `fine_tuned_model` |
| `--output-path` | 出力フォルダ（ファイル名は `{model名}.onnx`） | `fine_tuned_model/onnx` |
| `--input-size` | 320 または 512 | `512` |
| `--backbone-type` | `MLSD` (tiny) / `MLSD_large` | `MLSD_large` |
| `--batch-size` | エクスポート時バッチ | `1` |
| `--topk` | center 点の top-k | `200` |
| `--infer-from-ckpt` | ckpt 名から size / backbone を推定 | — |
| `--opset` | ONNX opset | auto |

---

## export_onnx_split.py

Split Computing 用に head / tail の 2 つの ONNX を出力します。主な設定はスクリプト内の `SplitExportConfig` です。

```bash
python onnx/export_onnx_split.py

python onnx/export_onnx_split.py \
  --checkpoint-dir fine_tuned_model/M-LSD_512_tiny_ft_100_00005 \
  --output-dir fine_tuned_model/onnx/split
```

`SplitExportConfig` の主な項目:

- `input_size`: 320 / 512
- `backbone_type`: `MLSD` / `MLSD_large`
- `split_point`: `block_1_project` / `block_3_project` / `block_6_project`
- `checkpoint_dir` / `output_dir`
- `run_validation`: head→tail の数値一致チェック（default: True）

出力例 (`M-LSD_512_tiny_split_block_1_project_*`):

- `{stem}_head.onnx`: 入力画像 → 中間特徴
- `{stem}_tail.onnx`: 中間特徴 → `org_disp_map` / `org_center_pts` / `org_center_scores`
- `{stem}_split_meta.json`: I/O 名・形状・圧縮率など

---

## validate_onnx.py

変換済み ONNX を対応する TF チェックポイントと比較します。

```bash
python onnx/validate_onnx.py \
  --checkpoint-dir fine_tuned_model/M-LSD_512_large_ft_50_000125 \
  --onnx-path fine_tuned_model/onnx/M-LSD_512_large_ft_50_000125.onnx

python onnx/validate_onnx.py \
  --checkpoint-dir .\fine_tuned_model\M-LSD_512_tiny_ft_100_00005\ \
  --onnx-path .\fine_tuned_model\onnx\M-LSD_512_tiny_ft_100_00005.onnx \
  --backbone-type MLSD
```

---

## analyze_onnx_tensor_sizes.py

ONNX グラフのテンソル要素数（numel）を調べ、入力より小さい出力を持つノードを列挙・要約します。Split の切れ目候補探しに使います。

解析と要約はこのスクリプト 1 本で完結します。  
モデル解析時はデフォルトで `onnx_analysis/{model_stem}_analysis.json` に保存します。

```bash
# 解析 + 要約（JSON も保存）
python onnx/analyze_onnx_tensor_sizes.py --model path/to/model.onnx

# 保存ファイル名だけ指定（保存先は常に onnx_analysis/）
python onnx/analyze_onnx_tensor_sizes.py --model path/to/model.onnx --save-json custom_name.json

# JSON を書かずに表示だけ
python onnx/analyze_onnx_tensor_sizes.py --model path/to/model.onnx --no-save-json

# 保存済み JSON の再要約
python onnx/analyze_onnx_tensor_sizes.py --json-path onnx_analysis/custom_name.json
```

要約に出る主な情報:

- 最初に「出力 numel < 入力 numel」になったノード
- 第1出力 shape ごとの代表ノード
- 出力が特に小さいノード
