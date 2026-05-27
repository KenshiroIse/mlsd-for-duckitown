"""
check_model_load.py

Purpose:
    Verify that the Keras checkpoint (ckpt_models/) is correctly loaded
    by comparing its outputs against the reference TFLite model.

What this script does:
    1. TFLite inference   — run the fp32 TFLite model on a test image and
                            inspect output shapes, displacement map scale,
                            and detected keypoints.
    2. Keras inference    — load the same checkpoint via load_model.py,
                            confirm that weights actually changed after
                            restore, then run inference on the same image.
    3. Scale comparison   — compare displacement-map standard deviations
                            between TFLite and Keras outputs.
                            A large ratio (>> 1x) indicates a failed restore.
    4. Stem kernel check  — print MobileNetV2 Conv1 kernel statistics to
                            confirm the backbone was restored from the
                            checkpoint (expected std ~0.01–0.1).
"""

import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
warnings.filterwarnings("ignore")

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from utils import _preprocess_lane_image

tf.get_logger().setLevel("ERROR")

# ── Path setup (parent.parent because this script runs inside fine-tuning/) ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "fine-tuning"))

TFLITE_PATH = PROJECT_ROOT / "tflite_models" / "M-LSD_512_large_fp32.tflite"
CKPT_DIR = PROJECT_ROOT / "ckpt_models" / "M-LSD_512_large"

# Test image (pick one from dataset/test/)
TEST_IMAGE = next((PROJECT_ROOT / "dataset" / "test").glob("*.jpg"), None)
if TEST_IMAGE is None:
    TEST_IMAGE = next((PROJECT_ROOT / "dataset" / "train").glob("*.jpg"))
print(f"Using image: {TEST_IMAGE}")

INPUT_SIZE = 512

# ── Common preprocessing ──────────────────────────────────────────────

raw_bgr = cv2.imread(str(TEST_IMAGE))
raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
proc_img, _ = _preprocess_lane_image(raw_rgb, [INPUT_SIZE, INPUT_SIZE])
# proc_img: [512, 512, 3]  RGB [0, 255] float/uint8

print(f"proc_img: shape={proc_img.shape}, dtype={proc_img.dtype}, min={proc_img.min():.1f}, max={proc_img.max():.1f}")

# ════════════════════════════════════════════════════════════
# 1. TFLite inference
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. TFLite inference")
print("=" * 60)

tflite_interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
tflite_interp.allocate_tensors()
tflite_in = tflite_interp.get_input_details()
tflite_out = tflite_interp.get_output_details()

print("Input  details:", [(d["name"], d["shape"], d["dtype"]) for d in tflite_in])
print("Output details:")
for i, d in enumerate(tflite_out):
    print(f"  [{i}] name={d['name']!r}  shape={d['shape']}")

rgba = np.concatenate([proc_img.astype("float32"), np.ones([INPUT_SIZE, INPUT_SIZE, 1], dtype="float32")], axis=-1)
batch = np.expand_dims(rgba, 0)
tflite_interp.set_tensor(tflite_in[0]["index"], batch)
tflite_interp.invoke()

raw_outputs = [tflite_interp.get_tensor(d["index"])[0] for d in tflite_out]
print("\nRaw output shapes:", [x.shape for x in raw_outputs])

# Identify output indices by shape
idx_vmap = next(i for i, x in enumerate(raw_outputs) if x.ndim == 3 and x.shape[-1] == 4)
idx_pts = next(i for i, x in enumerate(raw_outputs) if x.ndim == 2 and x.shape[-1] == 2)
idx_scores = next(i for i, x in enumerate(raw_outputs) if x.ndim == 1)

print(f"  vmap   → output[{idx_vmap}]  shape={raw_outputs[idx_vmap].shape}")
print(f"  pts    → output[{idx_pts}]  shape={raw_outputs[idx_pts].shape}")
print(f"  scores → output[{idx_scores}]  shape={raw_outputs[idx_scores].shape}")

tflite_vmap = raw_outputs[idx_vmap]
tflite_pts = raw_outputs[idx_pts]
tflite_scores = raw_outputs[idx_scores]
tflite_dist = np.sqrt(np.sum((tflite_vmap[:, :, :2] - tflite_vmap[:, :, 2:]) ** 2, axis=-1))

print(f"\nTFLite vmap:  min={tflite_vmap.min():.4f}  max={tflite_vmap.max():.4f}")
print(f"TFLite dist:  min={tflite_dist.min():.4f}  max={tflite_dist.max():.4f}  mean={tflite_dist.mean():.4f}")
high = tflite_scores > 0.1
if high.any():
    ys = tflite_pts[high, 0].astype(int)
    xs = tflite_pts[high, 1].astype(int)
    d = tflite_dist[ys, xs]
    print(f"TFLite score>0.1: {high.sum()} pts, dist at pts: min={d.min():.2f} max={d.max():.2f}")

# ════════════════════════════════════════════════════════════
# 2. Keras checkpoint inference (same preprocessing as TFLite)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Keras checkpoint inference (same preprocessing as TFLite)")
print("=" * 60)

import load_model

model_cfg = load_model.infer_config_from_path(str(CKPT_DIR))
model_cfg.batch_size = 1
model = load_model.build_model(model_cfg)

# Functional model: build once to create variables
dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])
_ = model(dummy, training=False)


def _find_decoder2_layer(model_obj):
    target_shape = (1, 1, 64, 16)
    for layer in model_obj.submodules:
        if isinstance(layer, tf.keras.layers.Conv2D) and tuple(layer.kernel.shape) == target_shape:
            return layer
    return None


backbone = model.get_layer("MLSD_large_extractor")
stem_conv = backbone.get_layer("Conv1")
decoder2_layer = _find_decoder2_layer(model)

stem_before = stem_conv.kernel.numpy().copy()
if decoder2_layer is not None:
    decoder2_before = decoder2_layer.kernel.numpy().copy()
    print(f"decoder2.kernel BEFORE restore: std={decoder2_before.std():.6f}  max={np.abs(decoder2_before).max():.6f}")
else:
    print("decoder2.kernel BEFORE restore: not found (skipping)")

load_model.load_checkpoint(model, str(CKPT_DIR))

_ = model(dummy, training=False)

stem_after = stem_conv.kernel.numpy()
stem_changed = not np.allclose(stem_before, stem_after)
print(f"Conv1.kernel AFTER  restore:  std={stem_after.std():.6f}  max={np.abs(stem_after).max():.6f}")
print(f"  → Weights changed: {stem_changed}")

if decoder2_layer is not None:
    decoder2_after = decoder2_layer.kernel.numpy()
    print(f"decoder2.kernel AFTER  restore:  std={decoder2_after.std():.6f}  max={np.abs(decoder2_after).max():.6f}")
    print(f"  → Weights changed: {not np.allclose(decoder2_before, decoder2_after)}")

# ── Run inference on the same proc_img ──
keras_input = tf.constant(proc_img[np.newaxis].astype("float32"))  # [1, 512, 512, 3]  RGB float32
keras_preds = model(keras_input, training=False)

print(f"\nKeras preds tuple length: {len(keras_preds)}")
for i, p in enumerate(keras_preds):
    print(f"  [{i}] shape={tuple(p.shape)}")

keras_vmap = keras_preds[10][0].numpy()
keras_pts = keras_preds[11][0].numpy()
keras_scores = keras_preds[12][0].numpy()
keras_dist = np.sqrt(np.sum((keras_vmap[:, :, :2] - keras_vmap[:, :, 2:]) ** 2, axis=-1))

print(f"\nKeras vmap:  min={keras_vmap.min():.4f}  max={keras_vmap.max():.4f}")
print(f"Keras dist:  min={keras_dist.min():.4f}  max={keras_dist.max():.4f}  mean={keras_dist.mean():.4f}")
k_high = keras_scores > 0.1
if k_high.any():
    ky = keras_pts[k_high, 0].astype(int)
    kx = keras_pts[k_high, 1].astype(int)
    kd = keras_dist[ky, kx]
    print(f"Keras score>0.1: {k_high.sum()} pts, dist at pts: min={kd.min():.2f} max={kd.max():.2f}")

# ════════════════════════════════════════════════════════════
# 3. Scale difference summary
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Summary")
print("=" * 60)
ratio = tflite_vmap.std() / (keras_vmap.std() + 1e-9)
print(f"TFLite vmap std: {tflite_vmap.std():.4f}")
print(f"Keras  vmap std: {keras_vmap.std():.4f}")
print(f"Scale ratio (TFLite/Keras): {ratio:.1f}x")
if ratio > 10:
    print("→ Weight loading likely failed (ratio >> 1)")
elif ratio < 2:
    print("→ Scale difference is small; preprocessing difference only")
else:
    print("→ Moderate difference; further investigation needed")

# ════════════════════════════════════════════════════════════
# 4. Additional: check the first Conv kernel of MobileNetV2
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. MobileNetV2 stem Conv1 kernel check")
print("=" * 60)
backbone = model.get_layer("MLSD_large_extractor")
stem_conv = backbone.get_layer("Conv1")
stem_w = stem_conv.depthwise_kernel.numpy() if hasattr(stem_conv, "depthwise_kernel") else stem_conv.kernel.numpy()
print(f"Backbone stem kernel: shape={stem_w.shape}  std={stem_w.std():.6f}  max={np.abs(stem_w).max():.6f}")
print("(ImageNet pretrained Conv1 kernel std is typically 0.01~0.1)")
