"""
Validate an existing ONNX model by comparing outputs with a TF checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import dataloader as dl
import tensorflow as tf
from load_model import ModelConfig, build_model, infer_config_from_path, load_checkpoint

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)

DEFAULT_IMAGE_DIR = os.path.join(repo_root, "dataset", "test")
DEFAULT_LABEL_PATH = os.path.join(DEFAULT_IMAGE_DIR, "_annotation.wireframe.json")


def _to_numpy(array_like):
    return array_like.numpy() if hasattr(array_like, "numpy") else np.asarray(array_like)


def decode_lines_from_model_output(
    preds,
    input_size: int,
    score_thr: float,
    dist_thr: float,
    debug: bool = False,
    debug_prefix: str = "",
) -> np.ndarray:
    pts = _to_numpy(preds[11][0])
    pts_score = _to_numpy(preds[12][0])
    vmap = _to_numpy(preds[10][0])

    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    if debug:
        print(f"{debug_prefix}dist_map: min={dist_map.min():.4f}, max={dist_map.max():.4f}, mean={dist_map.mean():.4f}")
        print(f"{debug_prefix}vmap(disp raw): min={vmap.min():.4f}, max={vmap.max():.4f}")
        high_score_mask = pts_score > score_thr
        if high_score_mask.any():
            ys = pts[high_score_mask, 0].astype(int)
            xs = pts[high_score_mask, 1].astype(int)
            dists_at_pts = dist_map[ys, xs]
            print(
                f"{debug_prefix}{high_score_mask.sum()} pts with score>{score_thr}: "
                f"dist min={dists_at_pts.min():.4f}, max={dists_at_pts.max():.4f}, "
                f"mean={dists_at_pts.mean():.4f}"
            )
            print(f"{debug_prefix}dist > {dist_thr}: {(dists_at_pts > dist_thr).sum()} pts pass")

    segments_list = []
    line_scores = []
    for (y, x), score in zip(pts, pts_score):
        y, x = int(y), int(x)
        distance = float(dist_map[y, x])
        if score > score_thr and distance > dist_thr:
            dx_s, dy_s, dx_e, dy_e = vmap[y, x]
            segments_list.append([x + dx_s, y + dy_s, x + dx_e, y + dy_e])
            line_scores.append(float(score))

    if not segments_list:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    lines = 2.0 * np.array(segments_list, dtype=np.float32)
    return lines, np.array(line_scores, dtype=np.float32)


def draw_lines_on_image(image_bgr: np.ndarray, lines: np.ndarray, color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    output = image_bgr.copy()
    if lines.size == 0:
        return output

    h, w = output.shape[:2]
    for x1, y1, x2, y2 in lines:
        x1 = np.clip(int(round(x1)), 0, w - 1)
        y1 = np.clip(int(round(y1)), 0, h - 1)
        x2 = np.clip(int(round(x2)), 0, w - 1)
        y2 = np.clip(int(round(y2)), 0, h - 1)
        cv2.line(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    return output


def load_labels(label_path: str, image_dir: str) -> tuple[list[str], list[list[list[float]]]]:
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label json not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    image_paths = []
    lines_list = []
    for item in labels:
        filename = item.get("filename")
        lines = item.get("lines", [])
        if not filename:
            continue
        image_paths.append(os.path.join(image_dir, filename))
        lines_list.append(lines)

    if not image_paths:
        raise ValueError("No labels found in label json.")

    return image_paths, lines_list


def resolve_output_tensors(
    onnx_outputs: list[np.ndarray], output_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_name = {name: out for name, out in zip(output_names, onnx_outputs)}

    org_disp_map = by_name.get("org_disp_map")
    org_center_pts = by_name.get("org_center_pts")
    org_center_scores = by_name.get("org_center_scores")

    for out in onnx_outputs:
        if org_disp_map is None and out.ndim == 4 and out.shape[-1] == 4:
            org_disp_map = out
        elif org_center_pts is None and out.ndim == 3 and out.shape[-1] == 2:
            org_center_pts = out
        elif org_center_scores is None and out.ndim == 2:
            org_center_scores = out

    if org_disp_map is None or org_center_pts is None or org_center_scores is None:
        shapes = [tuple(out.shape) for out in onnx_outputs]
        raise RuntimeError(f"Failed to resolve ONNX outputs (shapes={shapes}, names={output_names})")

    return org_disp_map, org_center_pts, org_center_scores


def build_decode_preds(
    org_disp_map: np.ndarray,
    org_center_pts: np.ndarray,
    org_center_scores: np.ndarray,
):
    preds = [None] * 13
    preds[10] = org_disp_map
    preds[11] = org_center_pts
    preds[12] = org_center_scores
    return preds


def tensor_diff_stats(name: str, tf_out: np.ndarray, onnx_out: np.ndarray) -> None:
    abs_diff = np.abs(tf_out - onnx_out)
    print(
        f"[parity] {name}: shape={tf_out.shape} "
        f"max={abs_diff.max():.6f} mean={abs_diff.mean():.6f} p99={np.percentile(abs_diff, 99):.6f}"
    )


def export_overlay_images(
    onnx_path: str,
    cfg: ModelConfig,
    ckpt_dir: str,
    image_dir: str,
    label_path: str,
    output_dir: str,
    score_thr: float,
    dist_thr: float,
    num_samples: int,
) -> None:

    os.makedirs(output_dir, exist_ok=True)
    image_paths, lines_list = load_labels(label_path, image_dir)
    if num_samples > 0:
        image_paths = image_paths[:num_samples]
        lines_list = lines_list[:num_samples]

    dataset = dl.build_dataloader(
        image_paths=image_paths,
        lines_list=lines_list,
        batch_size=1,
        target_size=cfg.input_size,
        augment=False,
    )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    tf_model = build_model(cfg)
    load_checkpoint(tf_model, ckpt_dir)
    tf_export_model = build_export_model(tf_model)

    print(f"[*] output dir: {output_dir}")
    print(f"[*] onnx output names: {output_names}")

    images_out_tf = []
    images_out = []
    line_counts = []
    tf_line_counts = []

    for idx, ((images, _targets), image_path) in enumerate(zip(dataset, image_paths), start=1):
        batch = images.numpy().astype(np.float32)
        image_u8 = batch[0].clip(0, 255).astype(np.uint8)

        tf_outputs = tf_export_model(batch, training=False)
        tf_org_disp_map = _to_numpy(tf_outputs[0])
        tf_org_center_pts = _to_numpy(tf_outputs[1])
        tf_org_center_scores = _to_numpy(tf_outputs[2])

        onnx_outputs = session.run(None, {input_name: batch})
        org_disp_map, org_center_pts, org_center_scores = resolve_output_tensors(onnx_outputs, output_names)
        tf_preds = build_decode_preds(tf_org_disp_map, tf_org_center_pts, tf_org_center_scores)
        onnx_preds = build_decode_preds(org_disp_map, org_center_pts, org_center_scores)

        is_first = len(images_out) == 0
        tf_lines, tf_line_scores = decode_lines_from_model_output(
            tf_preds,
            cfg.input_size,
            score_thr=score_thr,
            dist_thr=dist_thr,
            debug=is_first,
            debug_prefix="[tf] ",
        )
        lines, line_scores = decode_lines_from_model_output(
            onnx_preds,
            cfg.input_size,
            score_thr=score_thr,
            dist_thr=dist_thr,
            debug=is_first,
            debug_prefix="[onnx] ",
        )

        if is_first:
            tf_scores = _to_numpy(tf_preds[12][0])
            scores = _to_numpy(onnx_preds[12][0])
            print(
                f"  [tf]   scores: max={tf_scores.max():.4f}, "
                f">{score_thr}={(tf_scores > score_thr).sum()}, "
                f"lines_detected={len(tf_lines)}"
            )
            print(
                f"  [onnx] scores: max={scores.max():.4f}, "
                f">{score_thr}={(scores > score_thr).sum()}, "
                f"lines_detected={len(lines)}"
            )
            tensor_diff_stats("org_disp_map", tf_org_disp_map, org_disp_map)
            tensor_diff_stats("org_center_pts", tf_org_center_pts, org_center_pts)
            tensor_diff_stats("org_center_scores", tf_org_center_scores, org_center_scores)

        line_counts.append(len(lines))
        tf_line_counts.append(len(tf_lines))

        tf_overlay = draw_lines_on_image(cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR), tf_lines)
        overlay = draw_lines_on_image(cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR), lines)
        images_out_tf.append(tf_overlay)
        images_out.append(overlay)

        if len(images_out) >= num_samples:
            break

    print(f"TF lines per image: {tf_line_counts}")
    print(f"ONNX lines per image: {line_counts}")

    if images_out and images_out_tf:
        fig, axes = plt.subplots(2, len(images_out), figsize=(4 * len(images_out), 8))
        if len(images_out) == 1:
            axes = np.array(axes).reshape(2, 1)

        for col_idx, (tf_img, onnx_img) in enumerate(zip(images_out_tf, images_out)):
            axes[0, col_idx].imshow(cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB))
            axes[0, col_idx].axis("off")
            axes[0, col_idx].set_title(f"TF: {tf_line_counts[col_idx]} lines")

            axes[1, col_idx].imshow(cv2.cvtColor(onnx_img, cv2.COLOR_BGR2RGB))
            axes[1, col_idx].axis("off")
            axes[1, col_idx].set_title(f"ONNX: {line_counts[col_idx]} lines")

        fig.suptitle("TF vs ONNX Inference")
        plt.tight_layout()
        figure_path = os.path.join(output_dir, "tf_vs_onnx_inference_visualization.png")
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[*] visualization saved: {figure_path}")

    del tf_model
    tf.keras.backend.clear_session()


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
    parser.add_argument(
        "--output-dir",
        default=os.path.join(repo_root, "output"),
        help="Directory to save line-overlay images (default: ./output).",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.50,
        help="Score threshold used when decoding lines.",
    )
    parser.add_argument(
        "--dist-thr",
        type=float,
        default=5.0,
        help="Distance threshold used when decoding lines.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Maximum number of images to process for overlay export (<=0 means all).",
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
    print(f"[*] TF_USE_LEGACY_KERAS={os.environ.get('TF_USE_LEGACY_KERAS')}")

    validate_onnx(onnx_path, cfg, ckpt_dir)

    export_overlay_images(
        onnx_path=onnx_path,
        cfg=cfg,
        ckpt_dir=ckpt_dir,
        image_dir=DEFAULT_IMAGE_DIR,
        label_path=DEFAULT_LABEL_PATH,
        output_dir=args.output_dir,
        score_thr=args.score_thr,
        dist_thr=args.dist_thr,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
