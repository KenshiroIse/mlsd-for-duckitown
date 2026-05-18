"""
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
"""

import argparse
import sys
import os
from dataclasses import dataclass

import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)

from modules.models import WireFrameModel


@dataclass
class ModelConfig:
    input_size: int = 512
    map_size: int = 256
    backbone_type: str = "MLSD_large"
    pretrain: bool = True
    out_channel: int = 256
    dilate: int = 5
    final_last: bool = False
    final_act: bool = True
    final_res1: bool = False
    final_res2: bool = False
    residual_type: int = 0
    post_name: str = "_extractor"
    type_a_ksize: int = 1
    topk: int = 200
    final_padding_same: bool = True
    center_thr: float = 0.001
    batch_size: int = 1
    wd: float = 0.0001


def build_model(cfg: ModelConfig) -> tf.keras.Model:
    return WireFrameModel(cfg, training=False)


def load_checkpoint(model: tf.keras.Model, ckpt_dir: str) -> str:
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name="step"), model=model)
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=ckpt_dir,
        max_to_keep=3,
    )

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"[*] load ckpt from {manager.latest_checkpoint} at step {checkpoint.step.numpy()}.")
        return manager.latest_checkpoint

    print("[*] No checkpoint found. Using imagenet pretrained weights.")
    return ""


def load_pretrained_model(cfg: ModelConfig, ckpt_dir: str) -> tf.keras.Model:
    model = build_model(cfg)
    model.summary(line_length=80)
    load_checkpoint(model, ckpt_dir)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load pretrained M-LSD checkpoints for fine-tuning.")
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Checkpoint directory (e.g., ./ckpt_models/M-LSD_512_large).",
    )
    return parser.parse_args()


def infer_config_from_path(ckpt_dir: str) -> ModelConfig:
    model_name = os.path.basename(os.path.normpath(ckpt_dir))
    model_map = {
        "M-LSD_320_tiny": (320, "MLSD"),
        "M-LSD_320_large": (320, "MLSD_large"),
        "M-LSD_512_tiny": (512, "MLSD"),
        "M-LSD_512_large": (512, "MLSD_large"),
    }

    if model_name not in model_map:
        supported = ", ".join(sorted(model_map.keys()))
        raise ValueError(f"Unsupported checkpoint directory name: {model_name}. Supported: {supported}")

    input_size, backbone_type = model_map[model_name]
    return ModelConfig(
        input_size=input_size,
        map_size=input_size // 2,
        backbone_type=backbone_type,
        batch_size=1,
        topk=200,
    )


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt_dir}")

    cfg = infer_config_from_path(args.ckpt_dir)
    load_pretrained_model(cfg, args.ckpt_dir)


if __name__ == "__main__":
    main()
