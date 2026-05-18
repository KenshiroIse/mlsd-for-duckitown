import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="./dataset",
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument("--ignore-categories", default="", help="Comma-separated category names to ignore.")
    parser.add_argument("--min-area", type=float, default=50.0, help="Minimum polygon area (in pixels) to keep a mask.")
    parser.add_argument(
        "--approx-eps",
        type=float,
        default=2.0,
        help="Polyline simplification epsilon for approxPolyDP.",
    )
    parser.add_argument("--preview", action="store_true", help="Write overlay previews for visual inspection.")
    parser.add_argument("--preview-count", type=int, default=5, help="Total number of preview images to write.")
    return parser.parse_args()


def iter_polygons(segmentation):
    if not isinstance(segmentation, list):
        return
    for poly in segmentation:
        if not isinstance(poly, list):
            continue
        if len(poly) < 6:
            continue
        yield poly


def polygon_to_lines(poly, epsilon):
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return []
    if epsilon > 0:
        contour = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
    lines = []
    for idx in range(len(pts)):
        p1 = tuple(pts[idx])
        p2 = tuple(pts[(idx + 1) % len(pts)])
        if p1 == p2:
            continue
        lines.append((p1, p2))
    return lines


def draw_lines(img, lines, color=(0, 255, 0)):
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    root = args.root
    splits = ["train", "valid", "test"]
    ignore_set = {name.strip() for name in args.ignore_categories.split(",") if name.strip()}
    preview_written = 0

    if args.preview:
        preview_dir = os.path.join(root, "preview_wireframe")
        os.makedirs(preview_dir, exist_ok=True)

    for split in splits:
        coco_path = os.path.join(root, split, "_annotations.coco.json")
        images_dir = os.path.join(root, split)
        if not os.path.isfile(coco_path):
            print(f"Skip {split}: {coco_path} not found.")
            continue

        with open(coco_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
        images = {img["id"]: img for img in coco.get("images", [])}
        ann_by_image = defaultdict(list)
        for ann in coco.get("annotations", []):
            ann_by_image[ann["image_id"]].append(ann)

        total_images = 0
        kept_images = 0
        labels = []
        for image_id, img in images.items():
            total_images += 1
            file_name = img["file_name"]
            img_path = os.path.join(images_dir, file_name)
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            lines = []

            for ann in ann_by_image.get(image_id, []):
                cat_name = categories.get(ann["category_id"], "")
                if cat_name in ignore_set:
                    continue
                segs = list(iter_polygons(ann.get("segmentation", [])))
                if not segs:
                    continue

                for poly in segs:
                    contour = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    if cv2.contourArea(contour) < args.min_area:
                        continue
                    for p1, p2 in polygon_to_lines(poly, args.approx_eps):
                        lines.append([float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])])

            if not lines:
                continue

            labels.append({"filename": file_name, "lines": lines})
            kept_images += 1

            if args.preview and preview_written < args.preview_count:
                overlay = img_bgr.copy()
                draw_lines(overlay, lines)
                preview_name = f"{split}_{os.path.splitext(file_name)[0]}_preview.jpg"
                cv2.imwrite(os.path.join(preview_dir, preview_name), overlay)
                preview_written += 1

        label_path = os.path.join(root, split, "_annotation.wireframe.json")
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=True)

        print(f"Done: {split}")
        print(f"Total images: {total_images}")
        print(f"Images with lines: {kept_images}")
        print(f"Label json: {label_path}")
        if args.preview:
            print(f"preview images: {preview_written} saved under {preview_dir}")


if __name__ == "__main__":
    main()
