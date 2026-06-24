"""
Validate an existing ONNX model by comparing outputs with a TF checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)

from load_model import ModelConfig, build_model, infer_config_from_path, load_checkpoint


def resolve_checkpoint_dir(path: str) -> str:
    if os.path.isfile(path):
        raise ValueError(f"Checkpoint path must be a directory, got file: {path}")

    if os.path.exists(os.path.join(path, "checkpoint")):
        return path

    if not os.path.isdir(path):
        raise ValueError(f"Checkpoint path does not exist: {path}")

    candidates = []
    for name in os.listdir(path):
        subdir = os.path.join(path, name)
        if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "checkpoint")):
            candidates.append(subdir)

    if not candidates:
        raise ValueError(f"No checkpoint directory found under: {path}")

    return max(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "checkpoint")))


def build_config_from_args(args: argparse.Namespace, ckpt_dir: str) -> ModelConfig:
    if args.infer_from_ckpt:
        try:
            return infer_config_from_path(ckpt_dir)
        except ValueError:
            pass

    return ModelConfig(
        input_size=args.input_size,
        map_size=args.input_size // 2,
        backbone_type=args.backbone_type,
        batch_size=args.batch_size,
        topk=args.topk,
    )


def build_export_model(model: tf.keras.Model) -> tf.keras.Model:
    def passthrough(x):
        return x

    outputs = [
        tf.keras.layers.Lambda(passthrough, name="org_disp_map")(model.output[10]),
        tf.keras.layers.Lambda(passthrough, name="org_center_pts")(model.output[11]),
        tf.keras.layers.Lambda(passthrough, name="org_center_scores")(model.output[12]),
    ]
    return tf.keras.Model(model.input, outputs, name="WireFrameModel")


def validate_onnx(onnx_path: str, cfg: ModelConfig, ckpt_dir: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for validation") from exc

    model = build_model(cfg)
    load_checkpoint(model, ckpt_dir)
    export_model = build_export_model(model)

    dummy = np.random.rand(cfg.batch_size, cfg.input_size, cfg.input_size, 3).astype(np.float32)
    tf_outputs = export_model(dummy)
    tf_outputs = [out.numpy() for out in tf_outputs]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: dummy})

    for idx, (tf_out, ort_out) in enumerate(zip(tf_outputs, ort_outputs)):
        shape_match = tf_out.shape == ort_out.shape
        max_diff = float(np.max(np.abs(tf_out - ort_out))) if shape_match else None
        print(f"[validate] output[{idx}] shape tf={tf_out.shape} onnx={ort_out.shape} max_abs_diff={max_diff}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an existing ONNX model with a TF checkpoint")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Checkpoint directory or a parent directory containing checkpoints.",
    )
    parser.add_argument(
        "--onnx-path",
        required=True,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        choices=[320, 512],
        default=512,
        help="Input image size.",
    )
    parser.add_argument(
        "--backbone-type",
        choices=["MLSD", "MLSD_large"],
        default="MLSD_large",
        help="Backbone type.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for validation.")
    parser.add_argument("--topk", type=int, default=200, help="Top-k used in point selection.")
    parser.add_argument(
        "--infer-from-ckpt",
        action="store_true",
        help="Infer input size/backbone from checkpoint directory name when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    cfg = build_config_from_args(args, ckpt_dir)
    onnx_path = args.onnx_path

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print(f"[*] checkpoint: {ckpt_dir}")
    print(f"[*] onnx: {onnx_path}")
    print(f"[*] input size: {cfg.input_size}, backbone: {cfg.backbone_type}, topk: {cfg.topk}")

    validate_onnx(onnx_path, cfg, ckpt_dir)


if __name__ == "__main__":
    main()
