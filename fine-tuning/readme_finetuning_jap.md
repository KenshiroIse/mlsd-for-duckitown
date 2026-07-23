# fine-tuning README (Japanese)

学習・データ準備まわりのスクリプトまとめです。  
ONNX のエクスポート・検証・解析は [`../onnx/`](../onnx/) を参照してください。

データセット画像は **512×384**（4:3）です。推論側（app-lane-detection）のカメラ入力は **640×480**（同じく 4:3）で、前処理の縦横比は揃えています。

---

## convert_coco.py

COCO 形式アノテーションから線分ラベル JSON を作成します。

```bash
python fine-tuning/convert_coco.py --root ./dataset --preview
```

| 引数 | 説明 | default |
|---|---|---|
| `--root` | train/valid/test を含むデータセットルート | `./dataset` |
| `--ignore-categories` | 無視するカテゴリ名（カンマ区切り） | `""` |
| `--min-area` | ポリゴン面積の最小値 | `50.0` |
| `--approx-eps` | approxPolyDP の epsilon | `2.0` |
| `--preview` | プレビュー画像を書き出す | off |
| `--preview-count` | プレビュー枚数 | `8` |

出力:

- `{root}/{split}/_annotation.wireframe.json`
- `--preview` 時: `{root}/preview_wireframe/{split}/`

---

## dataloader.py

M-LSD 向け前処理と `tf.data` パイプライン。CLI はありません。

主な設定:

- `TARGET_SIZE`: 入力サイズ（512 または 320）
- `TP_SIGMA` / `JUNCTION_SIGMA`: ガウシアン sigma

前処理（app-lane-detection と同じ幾何）:

1. `top_cutoff = H // 3` で上をクロップ（**pad なし**）
2. ROI を `target_size × target_size` に bilinear resize
3. 線分ラベルも同じ幾何に合わせて変換

主な関数:

- `generate_target_maps_np(lines, height, width, ...)`: 16ch ターゲット生成
- `preprocess_image(image_path, lines, target_size)`: 上記前処理
- `load_data_wrapper(...)` / `build_dataloader(...)`: 学習用パイプライン

---

## dataloader_preview.py

前処理結果とターゲットマップのプレビューを書き出します。

```bash
python fine-tuning/dataloader_preview.py --target-size 512 --count 8
```

出力:

- `dataloader_preview/{target_size}/images`
- `dataloader_preview/{target_size}/targets`

---

## load_model.py

事前学習済みチェックポイントを読み込み、モデルを構築します。

```bash
python fine-tuning/load_model.py ./ckpt_models/M-LSD_512_large
```

対応名の例: `M-LSD_320_tiny` / `M-LSD_320_large` / `M-LSD_512_tiny` / `M-LSD_512_large`

---

## train_colab.ipynb / train.py

学習エントリポイント。画像前処理は `dataloader.build_dataloader` 経由のため、上記の幾何前処理がそのまま使われます。

Colab では Drive 上のリポジトリ（特に更新後の `dataloader.py`）が同期されていることを確認してください。

---

## check_model_load.py

TFLite / Keras チェックポイントの読み込みと簡易推論比較用の補助スクリプトです。

幾何前処理は `utils._preprocess_lane_image`（`H // 3` クロップ → 正方形 resize）を使います。
