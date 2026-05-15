"""
# paddiing(320)
cfg = DataConfig(input_size=320, mode="pad", grayscale=True)

# divding image into two images(320)
cfg = DataConfig(input_size=320, mode="split", grayscale=True)
"""

import json
import os
from dataclasses import dataclass, replace
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf


def _get_autotune() -> int:
    try:
        return tf.data.AUTOTUNE
    except AttributeError:
        return tf.data.experimental.AUTOTUNE


@dataclass
class DataConfig:
    root: str = "./dataset"
    split: str = "train"
    label_dir: str = "wireframe"
    input_size: int = 512
    base_height: int = 0
    crop_top: int = 0
    grayscale: bool = False
    mode: str = "pad"
    batch_size: int = 1
    shuffle: bool = True
    drop_remainder: bool = False
    num_parallel_calls: int = _get_autotune()


def _infer_base_height(input_size: int) -> int:
    if input_size == 320:
        return 240
    if input_size == 512:
        return 384
    return input_size


def _infer_crop_top(input_size: int) -> int:
    if input_size == 320:
        return 80
    if input_size == 512:
        return 128
    return 0


def _clip_line_to_rect(x1: float, y1: float, x2: float, y2: float, w: float, h: float):
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
    min_x, max_x = 0.0, w
    min_y, max_y = 0.0, h

    def _outcode(x, y):
        code = INSIDE
        if x < min_x:
            code |= LEFT
        elif x > max_x:
            code |= RIGHT
        if y < min_y:
            code |= BOTTOM
        elif y > max_y:
            code |= TOP
        return code

    code1 = _outcode(x1, y1)
    code2 = _outcode(x2, y2)

    while True:
        if not (code1 | code2):
            return x1, y1, x2, y2
        if code1 & code2:
            return None
        code_out = code1 if code1 else code2

        if code_out & TOP:
            if y2 == y1:
                return None
            x = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1)
            y = max_y
        elif code_out & BOTTOM:
            if y2 == y1:
                return None
            x = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1)
            y = min_y
        elif code_out & RIGHT:
            if x2 == x1:
                return None
            y = y1 + (y2 - y1) * (max_x - x1) / (x2 - x1)
            x = max_x
        else:
            if x2 == x1:
                return None
            y = y1 + (y2 - y1) * (min_x - x1) / (x2 - x1)
            x = min_x

        if code_out == code1:
            x1, y1 = x, y
            code1 = _outcode(x1, y1)
        else:
            x2, y2 = x, y
            code2 = _outcode(x2, y2)


def _transform_lines_padding(lines: np.ndarray, input_size: int, base_height: int, crop_top: int) -> np.ndarray:
    if lines.size == 0:
        return lines.reshape(0, 4)

    height = base_height - crop_top
    crop_offset = max(0, height - input_size)
    pad_top = 0 if height > input_size else input_size - height
    clip_height = min(height, input_size)
    output = []
    for x1, y1, x2, y2 in lines:
        y1 -= crop_top
        y2 -= crop_top
        y1 -= crop_offset
        y2 -= crop_offset
        clipped = _clip_line_to_rect(x1, y1, x2, y2, input_size, clip_height)
        if clipped is None:
            continue
        cx1, cy1, cx2, cy2 = clipped
        output.append((cx1, cy1 + pad_top, cx2, cy2 + pad_top))
    return np.array(output, dtype=np.float32)


def _transform_lines_split(lines: np.ndarray, input_size: int, base_height: int, crop_top: int):
    if lines.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    height = base_height - crop_top
    half = input_size // 2
    scale_x = input_size / float(half)
    scale_y = input_size / float(height) if height else 1.0

    left_out = []
    right_out = []
    for x1, y1, x2, y2 in lines:
        y1 -= crop_top
        y2 -= crop_top

        left_clip = _clip_line_to_rect(x1, y1, x2, y2, half, height)
        if left_clip is not None:
            lx1, ly1, lx2, ly2 = left_clip
            left_out.append((lx1 * scale_x, ly1 * scale_y, lx2 * scale_x, ly2 * scale_y))

        right_clip = _clip_line_to_rect(x1 - half, y1, x2 - half, y2, half, height)
        if right_clip is not None:
            rx1, ry1, rx2, ry2 = right_clip
            right_out.append((rx1 * scale_x, ry1 * scale_y, rx2 * scale_x, ry2 * scale_y))

    return np.array(left_out, dtype=np.float32), np.array(right_out, dtype=np.float32)


def load_labels(cfg: DataConfig) -> Tuple[List[str], List[List[List[float]]], List[int]]:
    label_path = os.path.join(cfg.root, cfg.label_dir, f"{cfg.split}.json")
    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    base_height = cfg.base_height or _infer_base_height(cfg.input_size)
    crop_top = cfg.crop_top or _infer_crop_top(cfg.input_size)

    filenames = []
    lines_list = []
    tile_indices = []

    for item in labels:
        filename = item.get("filename")
        lines = np.array(item.get("lines", []), dtype=np.float32).reshape(-1, 4)
        if not filename:
            continue

        if cfg.mode == "split":
            left, right = _transform_lines_split(lines, cfg.input_size, base_height, crop_top)
            if left.size:
                filenames.append(filename)
                lines_list.append(left.tolist())
                tile_indices.append(0)
            if right.size:
                filenames.append(filename)
                lines_list.append(right.tolist())
                tile_indices.append(1)
        else:
            padded = _transform_lines_padding(lines, cfg.input_size, base_height, crop_top)
            if padded.size:
                filenames.append(filename)
                lines_list.append(padded.tolist())
                tile_indices.append(-1)

    return filenames, lines_list, tile_indices


def _load_image(path: tf.Tensor, input_size: int, base_height: int, grayscale: bool) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [base_height, input_size], method="bilinear")
    if grayscale:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])
    image = tf.cast(image, tf.float32)
    return image


def build_targets(lines: tf.RaggedTensor, input_size: int):
    del input_size
    return lines


def _crop_and_pad(image: tf.Tensor, input_size: int, base_height: int, crop_top: int) -> tf.Tensor:
    image = image[crop_top:base_height, :, :]
    height = base_height - crop_top
    pad_top = input_size - height
    if pad_top < 0:
        image = image[-input_size:, :, :]
        pad_top = 0
    image = tf.pad(image, [[pad_top, 0], [0, 0], [0, 0]])
    return image


def _draw_lines(img: np.ndarray, lines: np.ndarray, color=(0, 255, 0)) -> None:
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)


def _crop_and_split(
    image: tf.Tensor, input_size: int, base_height: int, crop_top: int, tile_idx: tf.Tensor
) -> tf.Tensor:
    image = image[crop_top:base_height, :, :]
    half = input_size // 2
    left = image[:, :half, :]
    right = image[:, half:, :]
    tile = tf.cond(tf.equal(tile_idx, 0), lambda: left, lambda: right)
    return tf.image.resize(tile, [input_size, input_size], method="bilinear")


def build_dataset(cfg: DataConfig) -> tf.data.Dataset:
    label_path = os.path.join(cfg.root, cfg.label_dir, f"{cfg.split}.json")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label json not found: {label_path}")

    filenames, lines_list, tile_indices = load_labels(cfg)
    if not filenames:
        raise ValueError(f"No labels found in {label_path}")

    lines_rt = tf.ragged.constant(lines_list, dtype=tf.float32)
    tile_tensor = tf.constant(tile_indices, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, lines_rt, tile_tensor))

    if cfg.shuffle:
        dataset = dataset.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=True)

    base_height = cfg.base_height or _infer_base_height(cfg.input_size)
    crop_top = cfg.crop_top or _infer_crop_top(cfg.input_size)

    def _map_fn(filename: tf.Tensor, lines: tf.RaggedTensor, tile_idx: tf.Tensor):
        image_path = tf.strings.join([cfg.root, "/", cfg.split, "/", filename])
        image = _load_image(image_path, cfg.input_size, base_height, cfg.grayscale)
        if cfg.mode == "split":
            image = _crop_and_split(image, cfg.input_size, base_height, crop_top, tile_idx)
        else:
            image = _crop_and_pad(image, cfg.input_size, base_height, crop_top)
        targets = build_targets(lines, cfg.input_size)
        return {"image": image, "lines": targets, "filename": filename, "tile_idx": tile_idx}

    dataset = dataset.map(_map_fn, num_parallel_calls=cfg.num_parallel_calls)
    dataset = dataset.batch(cfg.batch_size, drop_remainder=cfg.drop_remainder)
    dataset = dataset.prefetch(_get_autotune())
    return dataset


def write_preprocess_preview(cfg: DataConfig, num_samples: int = 3, output_dir: str = "dataloader_preview") -> None:
    preview_cfg = replace(cfg, batch_size=1, shuffle=False, drop_remainder=False)
    dataset = build_dataset(preview_cfg)

    split_dir = os.path.join(output_dir, cfg.split)
    os.makedirs(split_dir, exist_ok=True)

    saved = 0
    for batch in dataset.take(num_samples):
        image = tf.squeeze(batch["image"], axis=0)
        lines_rt = batch["lines"]
        if hasattr(lines_rt, "values"):
            lines = lines_rt.values.numpy().reshape(-1, 4)
        else:
            lines = lines_rt.numpy().reshape(-1, 4)
        filename = batch["filename"].numpy()[0].decode("utf-8")
        tile_idx = int(batch["tile_idx"].numpy()[0])

        image = tf.clip_by_value(image, 0.0, 255.0)
        image_u8 = tf.cast(image, tf.uint8).numpy()
        overlay = image_u8.copy()
        if lines.size:
            _draw_lines(overlay, lines)

        base, ext = os.path.splitext(os.path.basename(filename))
        suffix = f"_tile{tile_idx}" if tile_idx >= 0 else ""
        out_name = f"{base}{suffix}{ext or '.png'}"
        out_path = os.path.join(split_dir, out_name)
        cv2.imwrite(out_path, overlay)
        saved += 1

    print(f"Preview saved: {saved} images -> {split_dir}")


if __name__ == "__main__":
    cfg = DataConfig()
    write_preprocess_preview(cfg)
