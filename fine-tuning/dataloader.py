import math
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

TARGET_SIZE = 512
TP_SIGMA = 1.0
JUNCTION_SIGMA = 1.0


def _get_autotune() -> int:
    try:
        return tf.data.AUTOTUNE
    except AttributeError:
        return tf.data.experimental.AUTOTUNE


def _draw_gaussian(map2d: np.ndarray, cx: float, cy: float, sigma: float) -> None:
    if sigma <= 0:
        return
    radius = max(1, int(3 * sigma))
    x0 = int(round(cx))
    y0 = int(round(cy))
    h, w = map2d.shape

    x_min = max(0, x0 - radius)
    x_max = min(w - 1, x0 + radius)
    y_min = max(0, y0 - radius)
    y_max = min(h - 1, y0 + radius)

    if x_min > x_max or y_min > y_max:
        return

    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    ys = np.arange(y_min, y_max + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    patch = map2d[y_min : y_max + 1, x_min : x_max + 1]
    np.maximum(patch, gaussian, out=patch)


def _stamp_window(map2d: np.ndarray, cx: int, cy: int, value: float, radius: int = 1) -> None:
    h, w = map2d.shape
    x_min = max(0, cx - radius)
    x_max = min(w - 1, cx + radius)
    y_min = max(0, cy - radius)
    y_max = min(h - 1, cy + radius)
    if x_min > x_max or y_min > y_max:
        return
    map2d[y_min : y_max + 1, x_min : x_max + 1] = value


def _sol_split_segments(
    x1: float, y1: float, x2: float, y2: float, mu: float
) -> List[Tuple[float, float, float, float]]:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return [(x1, y1, x2, y2)]

    k = int(math.floor(length / (mu / 2.0)) - 1)
    if k <= 1:
        return [(x1, y1, x2, y2)]

    ux = dx / length
    uy = dy / length
    seg_len = mu
    step = mu / 2.0
    segments = []
    for idx in range(k):
        start = idx * step
        end = min(length, start + seg_len)
        if end - start <= 1e-6:
            continue
        sx = x1 + ux * start
        sy = y1 + uy * start
        ex = x1 + ux * end
        ey = y1 + uy * end
        segments.append((sx, sy, ex, ey))
    return segments if segments else [(x1, y1, x2, y2)]


def generate_target_maps_np(
    lines: np.ndarray,
    height: int,
    width: int,
    tp_sigma: float = TP_SIGMA,
    junction_sigma: float = JUNCTION_SIGMA,
    line_width: int = 1,
) -> np.ndarray:
    if hasattr(lines, "numpy"):
        lines = lines.numpy()
    else:
        lines = np.asarray(lines)

    if hasattr(height, "numpy"):
        height = int(height.numpy())
    else:
        height = int(height)
    if hasattr(width, "numpy"):
        width = int(width.numpy())
    else:
        width = int(width)

    target = np.zeros((height, width, 16), dtype=np.float32)
    if lines.size == 0:
        return target

    lines = lines.reshape(-1, 4).astype(np.float32)

    diag = math.sqrt(width * width + height * height)
    mu = max(1.0, width * 0.125)

    def _accumulate_tp(x1: float, y1: float, x2: float, y2: float, base_ch: int) -> None:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        ix = int(round(cx))
        iy = int(round(cy))
        if not (0 <= ix < width and 0 <= iy < height):
            return

        _draw_gaussian(target[:, :, base_ch], cx, cy, tp_sigma)

        length = math.hypot(x2 - x1, y2 - y1)
        length_norm = min(1.0, length / (diag + 1e-6))
        angle = math.atan2(y2 - y1, x2 - x1)
        degree_norm = (angle / (2.0 * math.pi)) + 0.5

        _stamp_window(target[:, :, base_ch + 1], ix, iy, length_norm, radius=1)
        _stamp_window(target[:, :, base_ch + 2], ix, iy, degree_norm, radius=1)

        target[iy, ix, base_ch + 3] = x1 - cx
        target[iy, ix, base_ch + 4] = y1 - cy
        target[iy, ix, base_ch + 5] = x2 - cx
        target[iy, ix, base_ch + 6] = y2 - cy

    for x1, y1, x2, y2 in lines:
        _accumulate_tp(x1, y1, x2, y2, base_ch=0)

        line_map = np.ascontiguousarray(target[:, :, 15])
        cv2.line(
            line_map,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            1.0,
            max(1, int(line_width)),
            cv2.LINE_AA,
        )
        target[:, :, 15] = line_map

        _draw_gaussian(target[:, :, 14], x1, y1, junction_sigma)
        _draw_gaussian(target[:, :, 14], x2, y2, junction_sigma)

        for sx1, sy1, sx2, sy2 in _sol_split_segments(x1, y1, x2, y2, mu):
            _accumulate_tp(sx1, sy1, sx2, sy2, base_ch=7)

    return target


def preprocess_image(image_path: tf.Tensor, lines: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = image[128:, :, :]
    image = tf.pad(image, [[256, 0], [0, 0], [0, 0]])

    lines = tf.reshape(lines, [-1, 4])
    lines = tf.ensure_shape(lines, [None, 4])

    x1, y1, x2, y2 = tf.unstack(lines, axis=-1)
    y1_cropped = y1 - 128.0
    y2_cropped = y2 - 128.0
    keep = tf.logical_not(tf.logical_and(y1_cropped < 0.0, y2_cropped < 0.0))
    lines_cropped = tf.stack([x1, y1_cropped, x2, y2_cropped], axis=-1)
    lines_filtered = tf.boolean_mask(lines_cropped, keep)

    x1_f, y1_f, x2_f, y2_f = tf.unstack(lines_filtered, axis=-1)
    y1_padded = y1_f + 256.0
    y2_padded = y2_f + 256.0
    lines = tf.stack([x1_f, y1_padded, x2_f, y2_padded], axis=-1)

    if TARGET_SIZE == 320:
        scale = 320.0 / 512.0
        lines = lines * scale
        image = tf.image.resize(image, [320, 320], method="bilinear")

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, lines


def _flip_lines_horizontal(lines: tf.Tensor, width: int) -> tf.Tensor:
    x1, y1, x2, y2 = tf.unstack(lines, axis=-1)
    width_f = tf.cast(width, lines.dtype)
    x1_flipped = width_f - 1.0 - x1
    x2_flipped = width_f - 1.0 - x2
    return tf.stack([x1_flipped, y1, x2_flipped, y2], axis=-1)


def load_data_wrapper(
    image_path: tf.Tensor,
    lines: tf.RaggedTensor,
    target_size: int = TARGET_SIZE,
    augment: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    line_values = lines.values if hasattr(lines, "values") else lines
    line_values = tf.reshape(line_values, [-1, 4])
    image, line_values = preprocess_image(image_path, line_values)

    if augment:
        do_flip = tf.less(tf.random.uniform([]), 0.5)
        image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        line_values = tf.cond(
            do_flip,
            lambda: _flip_lines_horizontal(line_values, target_size),
            lambda: line_values,
        )

    scale = tf.cast(target_size // 2, tf.float32) / tf.cast(target_size, tf.float32)
    line_values = line_values * scale

    target = tf.py_function(
        generate_target_maps_np,
        inp=[line_values, tf.constant(target_size // 2, tf.int32), tf.constant(target_size // 2, tf.int32)],
        Tout=tf.float32,
    )
    target.set_shape([target_size // 2, target_size // 2, 16])
    image.set_shape([target_size, target_size, 3])
    return image, target


def build_dataloader(
    image_paths: List[str],
    lines_list: List[List[List[float]]],
    batch_size: int,
    target_size: int = TARGET_SIZE,
    augment: bool = False,
) -> tf.data.Dataset:
    lines_rt = tf.ragged.constant(lines_list, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, lines_rt))

    def _map_fn(image_path: tf.Tensor, lines: tf.RaggedTensor):
        return load_data_wrapper(image_path, lines, target_size=target_size, augment=augment)

    dataset = dataset.map(_map_fn, num_parallel_calls=_get_autotune())
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(_get_autotune())
    return dataset
