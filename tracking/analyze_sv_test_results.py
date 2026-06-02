import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SeqTrack SV test results and generate summary reports.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/silas/tracking/dataset/test_sv"),
        help="Path to test_sv dataset root.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/home/silas/tracking/algorithm/seqtrack3d/output/test/tracking_results/seqtrack/seqtrack_b384_3d"),
        help="Path to tracker result txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/silas/tracking/algorithm/seqtrack3d/output/test/analysis/seqtrack_b384_3d"),
        help="Directory to save summary json/csv/plots.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Analyze only one sequence (e.g. 01_000000). If omitted, analyze all available result files.",
    )
    parser.add_argument(
        "--plot_bin_gap",
        type=float,
        default=0.05,
        help="Threshold bin gap for success IoU curve.",
    )
    return parser.parse_args()


def read_xywh_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows: List[List[float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.replace("\t", ",")
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 4:
            raise ValueError(f"Invalid bbox row in {path}: {line}")
        rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])

    return np.asarray(rows, dtype=np.float64)


def read_time_file(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    values: List[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.replace("\t", ",")
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        values.append(float(parts[0]))
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def calc_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # xywh -> xyxy
    pred_x1 = pred[:, 0]
    pred_y1 = pred[:, 1]
    pred_x2 = pred[:, 0] + pred[:, 2]
    pred_y2 = pred[:, 1] + pred[:, 3]

    gt_x1 = gt[:, 0]
    gt_y1 = gt[:, 1]
    gt_x2 = gt[:, 0] + gt[:, 2]
    gt_y2 = gt[:, 1] + gt[:, 3]

    inter_x1 = np.maximum(pred_x1, gt_x1)
    inter_y1 = np.maximum(pred_y1, gt_y1)
    inter_x2 = np.minimum(pred_x2, gt_x2)
    inter_y2 = np.minimum(pred_y2, gt_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_pred = np.maximum(0.0, pred[:, 2]) * np.maximum(0.0, pred[:, 3])
    area_gt = np.maximum(0.0, gt[:, 2]) * np.maximum(0.0, gt[:, 3])
    union = np.maximum(area_pred + area_gt - inter, 1e-12)
    return inter / union


def calc_center_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_cx = pred[:, 0] + 0.5 * pred[:, 2]
    pred_cy = pred[:, 1] + 0.5 * pred[:, 3]
    gt_cx = gt[:, 0] + 0.5 * gt[:, 2]
    gt_cy = gt[:, 1] + 0.5 * gt[:, 3]
    return np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)


def calc_norm_center_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_cx = pred[:, 0] + 0.5 * pred[:, 2]
    pred_cy = pred[:, 1] + 0.5 * pred[:, 3]
    gt_cx = gt[:, 0] + 0.5 * gt[:, 2]
    gt_cy = gt[:, 1] + 0.5 * gt[:, 3]

    norm_w = np.maximum(gt[:, 2], 1e-12)
    norm_h = np.maximum(gt[:, 3], 1e-12)

    dx = (pred_cx - gt_cx) / norm_w
    dy = (pred_cy - gt_cy) / norm_h
    return np.sqrt(dx**2 + dy**2)


def robust_align_and_clean(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Align length
    if pred.shape[0] > gt.shape[0]:
        pred = pred[: gt.shape[0], :]
    elif pred.shape[0] < gt.shape[0]:
        pad = np.zeros((gt.shape[0] - pred.shape[0], 4), dtype=pred.dtype)
        pred = np.vstack([pred, pad])

    # Replace invalid zero-size predictions by previous frame prediction
    for i in range(1, pred.shape[0]):
        if (pred[i, 2] <= 0.0) or (pred[i, 3] <= 0.0):
            pred[i, :] = pred[i - 1, :]

    # Evaluation convention: first frame equals GT
    pred[0, :] = gt[0, :]

    valid = (gt[:, 2] > 0.0) & (gt[:, 3] > 0.0)
    return pred, gt, valid


def analyze_sequence(seq_name: str, dataset_root: Path, results_dir: Path, iou_thresholds: np.ndarray) -> Optional[Dict]:
    gt_path = dataset_root / seq_name / "groundTruth.rect"
    pred_path = results_dir / f"{seq_name}.txt"
    time_path = results_dir / f"{seq_name}_time.txt"

    if not gt_path.exists() or not pred_path.exists():
        return None

    gt = read_xywh_file(gt_path)
    pred = read_xywh_file(pred_path)
    pred, gt, valid = robust_align_and_clean(pred, gt)

    pred_v = pred[valid]
    gt_v = gt[valid]

    if pred_v.shape[0] == 0:
        return None

    iou = calc_iou(pred_v, gt_v)
    ce = calc_center_error(pred_v, gt_v)
    nce = calc_norm_center_error(pred_v, gt_v)

    success_curve = np.array([(iou > th).mean() for th in iou_thresholds], dtype=np.float64)
    auc = success_curve.mean()

    precision_20 = (ce <= 20.0).mean()
    norm_precision_02 = (nce <= 0.2).mean()

    t = read_time_file(time_path)
    fps = None
    if t is not None:
        t = t[t > 1e-12]
        if t.size > 0:
            fps = float(1.0 / t.mean())

    return {
        "sequence": seq_name,
        "num_frames_total": int(gt.shape[0]),
        "num_frames_valid": int(pred_v.shape[0]),
        "mean_iou": float(iou.mean()),
        "auc_success": float(auc),
        "precision_at_20px": float(precision_20),
        "norm_precision_at_0.2": float(norm_precision_02),
        "mean_center_error": float(ce.mean()),
        "median_center_error": float(np.median(ce)),
        "fps": fps,
        "success_curve": success_curve.tolist(),
    }


def collect_sequences(results_dir: Path, one_seq: Optional[str]) -> List[str]:
    if one_seq:
        return [one_seq]
    seqs: List[str] = []
    for p in sorted(results_dir.glob("*.txt")):
        if p.name.endswith("_time.txt"):
            continue
        seqs.append(p.stem)
    return seqs


def write_per_sequence_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "sequence",
        "num_frames_total",
        "num_frames_valid",
        "mean_iou",
        "auc_success",
        "precision_at_20px",
        "norm_precision_at_0.2",
        "mean_center_error",
        "median_center_error",
        "fps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, None) for k in fields})


def write_success_curve_csv(path: Path, thresholds: np.ndarray, curve: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iou_threshold", "success_rate"])
        for th, val in zip(thresholds.tolist(), curve.tolist()):
            writer.writerow([th, val])


def make_plots(rows: List[Dict], thresholds: np.ndarray, mean_curve: np.ndarray, out_dir: Path) -> None:
    if plt is None:
        print("[Warn] matplotlib unavailable, skip png plotting.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Success curve
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, mean_curve, color="tab:red", linewidth=2)
    plt.xlabel("IoU Threshold")
    plt.ylabel("Success Rate")
    plt.title("SV Test Success Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "success_curve.png", dpi=160)
    plt.close()

    # Sequence IoU distribution
    ious = np.array([r["mean_iou"] for r in rows], dtype=np.float64)
    plt.figure(figsize=(8, 5))
    plt.hist(ious, bins=20, color="tab:blue", alpha=0.8)
    plt.xlabel("Per-sequence Mean IoU")
    plt.ylabel("Count")
    plt.title("Sequence-level IoU Distribution")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "iou_histogram.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {args.dataset_root}")
    if not args.results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {args.results_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(0.0, 1.0 + 1e-9, args.plot_bin_gap, dtype=np.float64)
    sequences = collect_sequences(args.results_dir, args.sequence)
    if not sequences:
        raise RuntimeError(f"No result txt files in {args.results_dir}")

    rows: List[Dict] = []
    for seq_name in sequences:
        res = analyze_sequence(seq_name, args.dataset_root, args.results_dir, thresholds)
        if res is not None:
            rows.append(res)

    if not rows:
        raise RuntimeError("No valid sequence analyzed.")

    # Aggregate
    curves = np.array([r["success_curve"] for r in rows], dtype=np.float64)
    mean_curve = curves.mean(axis=0)

    mean_iou = float(np.mean([r["mean_iou"] for r in rows]))
    mean_auc = float(np.mean([r["auc_success"] for r in rows]))
    mean_prec20 = float(np.mean([r["precision_at_20px"] for r in rows]))
    mean_nprec02 = float(np.mean([r["norm_precision_at_0.2"] for r in rows]))
    mean_ce = float(np.mean([r["mean_center_error"] for r in rows]))

    fps_vals = [r["fps"] for r in rows if r["fps"] is not None]
    mean_fps = float(np.mean(fps_vals)) if fps_vals else None

    rows_sorted_iou = sorted(rows, key=lambda x: x["mean_iou"], reverse=True)
    top5 = [{"sequence": r["sequence"], "mean_iou": r["mean_iou"]} for r in rows_sorted_iou[:5]]
    bottom5 = [{"sequence": r["sequence"], "mean_iou": r["mean_iou"]} for r in rows_sorted_iou[-5:]]

    summary = {
        "dataset_root": str(args.dataset_root),
        "results_dir": str(args.results_dir),
        "num_sequences": len(rows),
        "num_frames_valid_total": int(sum(r["num_frames_valid"] for r in rows)),
        "mean_iou": mean_iou,
        "auc_success": mean_auc,
        "precision_at_20px": mean_prec20,
        "norm_precision_at_0.2": mean_nprec02,
        "mean_center_error": mean_ce,
        "mean_fps": mean_fps,
        "top5_by_iou": top5,
        "bottom5_by_iou": bottom5,
    }

    # Write outputs
    per_seq_csv = args.output_dir / "per_sequence_metrics.csv"
    curve_csv = args.output_dir / "success_curve.csv"
    summary_json = args.output_dir / "summary.json"

    write_per_sequence_csv(per_seq_csv, rows)
    write_success_curve_csv(curve_csv, thresholds, mean_curve)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    make_plots(rows, thresholds, mean_curve, args.output_dir)

    print("[Done] Analysis completed")
    print(f"  summary: {summary_json}")
    print(f"  per-sequence: {per_seq_csv}")
    print(f"  success-curve: {curve_csv}")
    if plt is not None:
        print(f"  plots: {args.output_dir / 'success_curve.png'}, {args.output_dir / 'iou_histogram.png'}")


if __name__ == "__main__":
    main()
