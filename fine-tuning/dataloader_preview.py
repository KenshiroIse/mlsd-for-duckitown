import argparse
import json
import os
import random
from typing import List, Tuple

import cv2
import dataloader as dl
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="./dataset/train", help="Image folder for the split.")
    parser.add_argument(
        "--label-path",
        default="./dataset/train/_annotation.wireframe.json",
        help="Label json path.",
    )
    parser.add_argument("--target-size", type=int, default=512, choices=[320, 512])
    parser.add_argument("--count", type=int, default=8, help="Number of preview images.")
    parser.add_argument("--output-dir", default="dataloader_preview", help="Output folder for previews.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before sampling previews.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for shuffle.")
    return parser.parse_args()


def load_labels(label_path: str, image_dir: str) -> List[Tuple[str, List[List[float]]]]:
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label json not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    items = []
    for item in labels:
        filename = item.get("filename")
        lines = item.get("lines", [])
        if not filename:
            continue
        items.append((os.path.join(image_dir, filename), lines))

    if not items:
        raise ValueError("No labels found in label json.")
    return items


def draw_lines(img: np.ndarray, lines: np.ndarray, color=(0, 255, 0)) -> None:
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)


def normalize_to_u8(map2d: np.ndarray) -> np.ndarray:
    if map2d.size == 0:
        return map2d.astype(np.uint8)
    min_v = float(np.min(map2d))
    max_v = float(np.max(map2d))
    if max_v <= min_v:
        return np.zeros_like(map2d, dtype=np.uint8)
    norm = (map2d - min_v) / (max_v - min_v)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def clip_lines_top_np(lines: np.ndarray, y_min: float) -> np.ndarray:
    if lines.size == 0:
        return lines.reshape(0, 4)

    lines = lines.reshape(-1, 4).astype(np.float32)
    output = []
    for x1, y1, x2, y2 in lines:
        if y1 < y_min and y2 < y_min:
            continue

        if y1 < y_min or y2 < y_min:
            denom = y2 - y1
            if abs(denom) < 1e-6:
                x_int = x1
            else:
                t = (y_min - y1) / denom
                x_int = x1 + t * (x2 - x1)

            if y1 < y_min:
                x1, y1 = x_int, y_min
            if y2 < y_min:
                x2, y2 = x_int, y_min

        output.append((x1, y1, x2, y2))

    return np.array(output, dtype=np.float32)


def preprocess_image_np(
    image_bgr: np.ndarray, lines: np.ndarray, target_size: int, crop_top=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Match app-lane-detection: H//3 crop (no pad), then square bilinear resize."""
    orig_h, orig_w = image_bgr.shape[:2]
    if crop_top is None:
        crop_top = orig_h // 3
    roi_h = max(orig_h - crop_top, 1)

    lines = clip_lines_top_np(lines, float(crop_top))

    scale_x = float(target_size) / float(max(orig_w, 1))
    scale_y = float(target_size) / float(roi_h)
    if lines.size:
        lines = lines.astype(np.float32, copy=False)
        lines[:, 0] *= scale_x
        lines[:, 2] *= scale_x
        lines[:, 1] = (lines[:, 1] - float(crop_top)) * scale_y
        lines[:, 3] = (lines[:, 3] - float(crop_top)) * scale_y

    image_bgr = image_bgr[crop_top:, :, :]
    image_bgr = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    image_bgr = image_bgr.astype(np.float32)  # Keep in [0, 255]

    return image_bgr, lines


def main() -> None:
    args = parse_args()
    items = load_labels(args.label_path, args.image_dir)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(items)

    output_dir = os.path.join(args.output_dir, str(args.target_size))
    images_dir = os.path.join(output_dir, "images")
    targets_dir = os.path.join(output_dir, "targets")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)

    saved = 0
    for image_path, lines in items:
        if saved >= args.count:
            break

        if not os.path.isfile(image_path):
            continue

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        lines_np = np.asarray(lines, dtype=np.float32)
        image_bgr, lines_np = preprocess_image_np(image_bgr, lines_np, args.target_size)
        if lines_np.size:
            draw_lines(image_bgr, lines_np)

        image_bgr = image_bgr.clip(0.0, 255.0).astype(np.uint8)

        base = os.path.splitext(os.path.basename(image_path))[0]
        out_name = f"{base}_preview_{args.target_size}.jpg"
        out_path = os.path.join(images_dir, out_name)
        cv2.imwrite(out_path, image_bgr)
        saved += 1

    image_paths = [path for path, _ in items]
    lines_list = [lines for _, lines in items]
    dataset = dl.build_dataloader(
        image_paths=image_paths,
        lines_list=lines_list,
        batch_size=1,
        target_size=args.target_size,
        augment=False,
    )

    for images, targets in dataset.take(1):
        image = images.numpy()[0]
        image_u8 = image.clip(0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(targets_dir, "sample_image.png"), image_bgr)

        target = targets.numpy()[0]
        for ch in range(target.shape[-1]):
            out = normalize_to_u8(target[:, :, ch])
            cv2.imwrite(os.path.join(targets_dir, f"target_ch{ch:02d}.png"), out)

    print(f"Preview saved: {saved} images -> {images_dir}")
    print(f"Target maps saved -> {targets_dir}")


if __name__ == "__main__":
    main()
