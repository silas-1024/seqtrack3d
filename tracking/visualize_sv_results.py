import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SeqTrack results on SV test dataset.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/silas/tracking/dataset/test_sv"),
        help="Path to test_sv dataset root.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/home/silas/tracking/algorithm/seqtrack_3d/output/test/tracking_results/seqtrack/seqtrack_b384_3d"),
        help="Path to tracker result txt directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/silas/tracking/algorithm/seqtrack_3d/output/test/visualizations/seqtrack_b384_3d"),
        help="Directory to store rendered videos.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Single sequence name like 01_000000. If omitted, process all available results.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Render only first N frames for quick preview. 0 means all frames.",
    )
    parser.add_argument(
        "--draw_gt",
        action="store_true",
        help="If set, draw ground-truth boxes from groundTruth.rect.",
    )
    return parser.parse_args()


def read_boxes(txt_path: Path) -> np.ndarray:
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing box file: {txt_path}")

    lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    rows: List[List[float]] = []
    for line in lines:
        line = line.replace("\t", ",")
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 4:
            raise ValueError(f"Invalid bbox line in {txt_path}: {line}")
        rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])

    return np.asarray(rows, dtype=np.float32)


def sorted_frames(seq_dir: Path) -> List[Path]:
    frames = [p for p in seq_dir.iterdir() if p.is_file()]
    frames.sort(key=lambda p: p.name)
    return frames


def xywh_to_xyxy(box: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box.tolist()
    x1 = int(round(max(0.0, x)))
    y1 = int(round(max(0.0, y)))
    x2 = int(round(min(float(width - 1), x + w)))
    y2 = int(round(min(float(height - 1), y + h)))
    return x1, y1, x2, y2


def render_sequence(
    seq_name: str,
    dataset_root: Path,
    results_dir: Path,
    output_dir: Path,
    fps: int,
    max_frames: int,
    draw_gt: bool,
) -> Optional[Path]:
    seq_root = dataset_root / seq_name
    frame_dir = seq_root / "sequences"
    pred_path = results_dir / f"{seq_name}.txt"
    gt_path = seq_root / "groundTruth.rect"

    if not frame_dir.exists():
        print(f"[Skip] sequence frames missing: {frame_dir}")
        return None
    if not pred_path.exists():
        print(f"[Skip] prediction file missing: {pred_path}")
        return None

    frame_files = sorted_frames(frame_dir)
    if not frame_files:
        print(f"[Skip] empty frame list: {frame_dir}")
        return None

    pred_boxes = read_boxes(pred_path)
    gt_boxes = read_boxes(gt_path) if draw_gt and gt_path.exists() else None

    total_frames = min(len(frame_files), len(pred_boxes))
    if gt_boxes is not None:
        total_frames = min(total_frames, len(gt_boxes))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    if total_frames <= 0:
        print(f"[Skip] no valid frame for {seq_name}")
        return None

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"[Skip] failed to read first frame: {frame_files[0]}")
        return None

    h, w = first_frame.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{seq_name}.mp4"

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path}")

    for idx in range(total_frames):
        frame = cv2.imread(str(frame_files[idx]))
        if frame is None:
            continue

        px1, py1, px2, py2 = xywh_to_xyxy(pred_boxes[idx], w, h)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(frame, "Pred", (px1, max(15, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if gt_boxes is not None:
            gx1, gy1, gx2, gy2 = xywh_to_xyxy(gt_boxes[idx], w, h)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(frame, "GT", (gx1, max(15, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(
            frame,
            f"{seq_name}  frame:{idx + 1}/{total_frames}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    writer.release()
    print(f"[OK] {seq_name}: {out_path}")
    return out_path


def collect_sequences(results_dir: Path, specified: Optional[str]) -> List[str]:
    if specified:
        return [specified]

    seqs: List[str] = []
    for p in sorted(results_dir.glob("*.txt")):
        if p.name.endswith("_time.txt"):
            continue
        seqs.append(p.stem)
    return seqs


def main() -> None:
    args = parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {args.dataset_root}")
    if not args.results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {args.results_dir}")

    sequences = collect_sequences(args.results_dir, args.sequence)
    if not sequences:
        raise RuntimeError(f"No result txt found in {args.results_dir}")

    ok = 0
    for seq in sequences:
        out = render_sequence(
            seq_name=seq,
            dataset_root=args.dataset_root,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            draw_gt=args.draw_gt,
        )
        if out is not None:
            ok += 1

    print(f"Finished. Rendered {ok}/{len(sequences)} sequences. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
