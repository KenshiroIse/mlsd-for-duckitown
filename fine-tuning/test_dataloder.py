import json
import os

import cv2
import dataloader as dl
import numpy as np

ROOT = "./dataset"
SPLIT = "train"
OUTPUT_DIR = "preview_dataloader"


def _normalize_to_u8(map2d: np.ndarray) -> np.ndarray:
    if map2d.size == 0:
        return map2d.astype(np.uint8)
    min_v = float(np.min(map2d))
    max_v = float(np.max(map2d))
    if max_v <= min_v:
        return np.zeros_like(map2d, dtype=np.uint8)
    norm = (map2d - min_v) / (max_v - min_v)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def _load_labels(root: str, split: str):
    label_path = os.path.join(root, split, "_annotation.wireframe.json")
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
        image_paths.append(os.path.join(root, split, filename))
        lines_list.append(lines)
    return image_paths, lines_list


def main() -> None:
    image_paths, lines_list = _load_labels(ROOT, SPLIT)
    if not image_paths:
        raise ValueError("No labels found. Check dataset and label_dir.")

    dataset = dl.build_dataloader(
        image_paths=image_paths,
        lines_list=lines_list,
        batch_size=1,
        target_size=dl.TARGET_SIZE,
        augment=False,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for images, targets in dataset.take(1):
        image = images.numpy()[0]
        image_u8 = (image * 255.0).clip(0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "sample_image.png"), image_bgr)

        target = targets.numpy()[0]
        for ch in range(target.shape[-1]):
            out = _normalize_to_u8(target[:, :, ch])
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"target_ch{ch:02d}.png"), out)

    print(f"Saved preview to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
