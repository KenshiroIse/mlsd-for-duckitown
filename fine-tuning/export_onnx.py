"""
Export M-LSD TensorFlow checkpoints to ONNX.

Default behavior targets fine-tuned checkpoints and exports the three outputs
used in frozen_models.py: center points, center scores, and displacement map.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_ONNX_OPSET = 13

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
import tf2onnx

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


def export_onnx(
    cfg: ModelConfig,
    ckpt_dir: str,
    output_path: str,
) -> tf.keras.Model:
    model = build_model(cfg)
    load_checkpoint(model, ckpt_dir)
    export_model = build_export_model(model)

    input_signature = (
        tf.TensorSpec(
            (cfg.batch_size, cfg.input_size, cfg.input_size, 3),
            tf.float32,
            name="input_image",
        ),
    )

    tf2onnx.convert.from_keras(
        export_model,
        input_signature=input_signature,
        opset=DEFAULT_ONNX_OPSET,
        output_path=output_path,
    )

    return export_model


def parse_args() -> argparse.Namespace:
    default_ckpt = os.path.join(repo_root, "fine_tuned_model")
    parser = argparse.ArgumentParser(description="Export M-LSD TensorFlow checkpoints to ONNX")
    parser.add_argument(
        "--checkpoint-dir",
        default=default_ckpt,
        help="Checkpoint directory or a parent directory containing checkpoints.",
    )
    parser.add_argument(
        "--output-path",
        default=default_ckpt + "/onnx",
        help="Directory to write the ONNX model (default: fine_tuned_model/onnx).",
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export.")
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

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    ckpt_name = os.path.basename(os.path.normpath(ckpt_dir))
    output_path = os.path.join(output_dir, f"{ckpt_name}.onnx")

    print(f"[*] checkpoint: {ckpt_dir}")
    print(f"[*] output: {output_path}")
    print(f"[*] input size: {cfg.input_size}, backbone: {cfg.backbone_type}, topk: {cfg.topk}")
    print(f"[*] opset: {DEFAULT_ONNX_OPSET}")

    export_onnx(cfg, ckpt_dir, output_path)
    print("[*] export complete")


if __name__ == "__main__":
    main()
