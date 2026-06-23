#!/usr/bin/env python3
"""Precompute CPU ORB/RANSAC affine transforms for video sequences."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MOTION_SPEC = importlib.util.spec_from_file_location(
    "seqtrack_global_motion", PROJECT_ROOT / "lib" / "utils" / "global_motion.py")
_MOTION_MODULE = importlib.util.module_from_spec(_MOTION_SPEC)
_MOTION_SPEC.loader.exec_module(_MOTION_MODULE)
IDENTITY_AFFINE = _MOTION_MODULE.IDENTITY_AFFINE
estimate_affine_motion = _MOTION_MODULE.estimate_affine_motion


IMAGE_DIR_NAMES = ("sequences", "img", "images", "color")
ANNO_NAMES = (
    "groundTruth.rect", "groundtruth.txt", "groundtruth_rect.txt",
    "new_groundtruth.txt", "groundtruth_skip.txt", "groundTruth.txt",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument(
        "--image-suffix", default=".jpg,.jpeg,.png,.tif,.tiff",
        help="Comma-separated image suffixes.")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--frame-id-mode", choices=("index", "stem"), default="index",
        help="Use zero-based sequence indices (tracker convention) or file stems.")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--max-reproj-error", type=float, default=5.0)
    return parser.parse_args()


def find_sequences(root, suffixes):
    sequences = []
    for directory, _, files in os.walk(root):
        images = sorted(
            Path(directory) / name for name in files
            if Path(name).suffix.lower() in suffixes)
        if not images:
            continue
        path = Path(directory)
        if path.name in IMAGE_DIR_NAMES:
            seq_dir = path.parent
        else:
            seq_dir = path
        sequences.append((seq_dir.name, seq_dir, images))
    return sequences


def read_boxes(seq_dir, count):
    anno_path = next((seq_dir / name for name in ANNO_NAMES
                      if (seq_dir / name).is_file()), None)
    if anno_path is None:
        return None
    try:
        boxes = np.genfromtxt(anno_path, delimiter=",", dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes[None]
        return boxes[:count, :4] if boxes.shape[1] >= 4 else None
    except Exception:
        return None


def quality_ok(affine, stats, min_inlier_ratio, max_reproj_error):
    linear = affine[:, :2].astype(np.float64)
    determinant = abs(np.linalg.det(linear))
    singular_values = np.linalg.svd(linear, compute_uv=False)
    return (
        stats["success"] > 0
        and stats["inlier_ratio"] >= min_inlier_ratio
        and stats["reprojection_error"] <= max_reproj_error
        and 0.25 <= determinant <= 4.0
        and singular_values.min() >= 0.25
        and singular_values.max() <= 4.0
    )


def process_sequence(task):
    (seq_name, seq_dir, images, output_path, overwrite, frame_id_mode,
     min_inlier_ratio, max_reproj_error) = task
    if output_path.exists() and not overwrite:
        return seq_name, "skipped", None

    boxes = read_boxes(seq_dir, len(images))
    affines, valid, ratios, errors, fallback = [], [], [], [], []
    prev_bgr = cv2.imread(str(images[0]))
    prev = None if prev_bgr is None else cv2.cvtColor(
        prev_bgr, cv2.COLOR_BGR2RGB)
    for idx in range(len(images) - 1):
        curr_bgr = cv2.imread(str(images[idx + 1]))
        if curr_bgr is None or prev is None:
            affine = IDENTITY_AFFINE.copy()
            stats = {"success": 0.0, "inlier_ratio": 0.0,
                     "reprojection_error": 0.0}
            curr = None if curr_bgr is None else cv2.cvtColor(
                curr_bgr, cv2.COLOR_BGR2RGB)
        else:
            curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
            prev_box = boxes[idx] if boxes is not None and idx < len(boxes) else None
            curr_box = boxes[idx + 1] if boxes is not None and idx + 1 < len(boxes) else None
            affine, stats = estimate_affine_motion(
                prev, curr, prev_bbox=prev_box, curr_bbox=curr_box)

        accepted = quality_ok(
            affine, stats, min_inlier_ratio, max_reproj_error)
        affines.append(affine if accepted else IDENTITY_AFFINE.copy())
        valid.append(accepted)
        ratios.append(stats["inlier_ratio"])
        errors.append(stats["reprojection_error"])
        fallback.append(not accepted)
        prev = curr

    if frame_id_mode == "stem":
        frame_ids = np.asarray([image.stem for image in images])
    else:
        frame_ids = np.arange(len(images), dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_path,
        affines=np.asarray(affines, dtype=np.float32).reshape(-1, 2, 3),
        valid=np.asarray(valid, dtype=bool),
        inlier_ratio=np.asarray(ratios, dtype=np.float32),
        reproj_error=np.asarray(errors, dtype=np.float32),
        fallback_identity=np.asarray(fallback, dtype=bool),
        frame_ids=frame_ids,
    )
    os.replace(temp_path, output_path)
    successful = np.asarray(valid, dtype=bool)
    summary = {
        "pairs": len(valid),
        "success": float(successful.mean()) if len(valid) else 1.0,
        "fallback": float(np.mean(fallback)) if fallback else 0.0,
        "inlier": float(np.mean(np.asarray(ratios)[successful]))
        if successful.any() else 0.0,
        "reproj": float(np.mean(np.asarray(errors)[successful]))
        if successful.any() else 0.0,
    }
    return seq_name, "written", summary


def main():
    args = parse_args()
    if cv2 is None:
        raise SystemExit(
            "OpenCV is required for affine precomputation. "
            "Install opencv-python or opencv-python-headless.")
    cv2.setNumThreads(1)
    root = Path(args.dataset_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root \
        else root / "affine_cache"
    suffixes = {suffix.strip().lower() for suffix in args.image_suffix.split(",")}
    sequences = find_sequences(root, suffixes)
    if not sequences:
        raise SystemExit(f"No image sequences found under {root}")

    tasks = []
    for seq_name, seq_dir, images in sequences:
        output = cache_root / args.dataset_name / f"{seq_name}.npz"
        tasks.append((
            seq_name, seq_dir, images, output, args.overwrite,
            args.frame_id_mode, args.min_inlier_ratio, args.max_reproj_error))

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process_sequence, task) for task in tasks]
        for future in as_completed(futures):
            seq_name, state, summary = future.result()
            if summary is None:
                print(f"[{seq_name}] skipped (cache exists)")
            else:
                print(
                    f"[{seq_name}] pairs={summary['pairs']} "
                    f"success={summary['success']:.2%} "
                    f"fallback={summary['fallback']:.2%} "
                    f"mean_inlier_ratio={summary['inlier']:.4f} "
                    f"mean_reprojection_error={summary['reproj']:.4f}")


if __name__ == "__main__":
    main()
