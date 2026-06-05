"""
ONNX evaluation for Phase 4 (Colab + local).

- Splits: valid, test (images on Google Drive → DATA_DIR/raw in Colab)
- Metrics: mAP, AP50, AP75, AR, per-class AP (pycocotools)
- Plots + JSON report saved to report_dir (Drive checkpoints/eval_reports)
- FiftyOne visual QA: src/fiftyone_onnx_review.py (Cell 19)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.inference import create_session, postprocess_to_coco_results, preprocess_image_rgb

DEFAULT_SPLITS = ("valid", "test")
ANN_FILENAME = "_annotations.coco.json"

COCO_METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def resolve_split_dir(raw_dir: str, split: str) -> str:
    """Image folder for a split (Roboflow flat or split/images/)."""
    images_subdir = os.path.join(raw_dir, split, "images")
    if os.path.isdir(images_subdir):
        return images_subdir
    return os.path.join(raw_dir, split)


def resolve_ann_path(raw_dir: str, split: str, ann_filename: str = ANN_FILENAME) -> str:
    return os.path.join(raw_dir, split, ann_filename)


def assert_onnx_artifacts(*paths: str) -> None:
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing ONNX file(s):\n  " + "\n  ".join(missing))


def metrics_from_evaluator(evaluator: COCOeval) -> dict[str, float]:
    stats = evaluator.stats
    return {
        name: float(stats[i]) if i < len(stats) else 0.0
        for i, name in enumerate(COCO_METRIC_NAMES)
    }


def per_class_ap(evaluator: COCOeval, coco_gt: COCO) -> dict[str, float]:
    precision = evaluator.eval.get("precision")
    if precision is None:
        return {}

    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    aind = [i for i, aRng in enumerate(evaluator.params.areaRng) if aRng == [0, 10000000000.0]]
    mind = [i for i, mDet in enumerate(evaluator.params.maxDets) if mDet == 100]
    if not aind or not mind:
        return {}

    a_idx, m_idx = aind[0], mind[0]
    result: dict[str, float] = {}
    for k, cat_id in enumerate(evaluator.params.catIds):
        name = cats.get(cat_id, str(cat_id))
        s = precision[:, :, k, a_idx, m_idx]
        s = s[s > -1]
        result[name] = float(np.mean(s)) if s.size else 0.0
    return result


def run_coco_segm_eval(
    gt_file: str,
    results: list[dict[str, Any]],
) -> tuple[COCOeval, COCO]:
    coco_gt = COCO(gt_file)
    if not results:
        coco_dt = coco_gt.loadRes([])
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            pred_file = f.name
        try:
            coco_dt = coco_gt.loadRes(pred_file)
        finally:
            os.unlink(pred_file)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator, coco_gt


def collect_split_predictions(
    session,
    raw_dir: str,
    split: str,
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
    ann_filename: str = ANN_FILENAME,
) -> tuple[list[dict[str, Any]], str, int, str]:
    """Run ONNX on one split. Returns (coco_results, ann_path, n_images, img_dir)."""
    ann_path = resolve_ann_path(raw_dir, split, ann_filename)
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Missing annotations for split '{split}': {ann_path}")

    img_dir = resolve_split_dir(raw_dir, split)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing image dir for split '{split}': {img_dir}")

    coco = COCO(ann_path)
    img_ids = sorted(coco.imgs.keys())
    if max_samples is not None:
        img_ids = img_ids[:max_samples]

    results: list[dict[str, Any]] = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor, orig_h, orig_w = preprocess_image_rgb(img_rgb, img_size)
        masks_logits, class_logits = session.run(None, {"pixel_values": tensor})
        results.extend(
            postprocess_to_coco_results(
                masks_logits,
                class_logits,
                orig_h,
                orig_w,
                image_id=img_id,
                score_threshold=score_threshold,
            )
        )

    return results, ann_path, len(img_ids), img_dir


def evaluate_split(
    session,
    raw_dir: str,
    split: str,
    model_tag: str,
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Full metrics for one model on one split."""
    results, ann_path, n_images, img_dir = collect_split_predictions(
        session,
        raw_dir,
        split,
        img_size=img_size,
        score_threshold=score_threshold,
        max_samples=max_samples,
    )
    evaluator, coco_gt = run_coco_segm_eval(ann_path, results)
    metrics = metrics_from_evaluator(evaluator)
    class_ap = per_class_ap(evaluator, coco_gt)

    info = {
        "model": model_tag,
        "split": split,
        "mAP": metrics["AP"],
        "metrics": metrics,
        "per_class_ap": class_ap,
        "n_images": n_images,
        "n_predictions": len(results),
        "ann_path": ann_path,
        "img_dir": img_dir,
        "avg_preds_per_image": len(results) / n_images if n_images else 0.0,
    }
    print(
        f"  [{split}/{model_tag}] images={n_images} preds={len(results)} "
        f"AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f} AP75={metrics['AP75']:.4f}"
    )
    return info


def _print_summary_table(summary: dict[str, dict[str, dict[str, Any]]], splits: list[str]) -> None:
    print("\n=== Metrics summary (valid / test) ===")
    header = f"{'split':<8} {'model':<6} {'AP':>7} {'AP50':>7} {'AP75':>7} {'AR100':>7} {'delta':>7}"
    print(header)
    print("-" * len(header))
    for split in splits:
        fp32 = summary[split]["fp32"]
        int8 = summary[split]["int8"]
        for tag, info in (("fp32", fp32), ("int8", int8)):
            m = info["metrics"]
            delta = ""
            if tag == "int8":
                delta = f"{info['mAP'] - fp32['mAP']:+.4f}"
            print(
                f"{split:<8} {tag:<6} {m['AP']:7.4f} {m['AP50']:7.4f} "
                f"{m['AP75']:7.4f} {m['AR_100']:7.4f} {delta:>7}"
            )


def save_report_json(summary: dict[str, Any], report_dir: str) -> str:
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "phase4_eval_report.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Report JSON → {path}")
    return path


def plot_phase4_report(summary: dict[str, Any], report_dir: str) -> list[str]:
    """Save comparison plots. Returns list of PNG paths."""
    import matplotlib.pyplot as plt

    os.makedirs(report_dir, exist_ok=True)
    splits = summary["splits"]
    data = summary["results"]
    saved: list[str] = []

    # 1) mAP / AP50 / AP75 by split (fp32 vs int8)
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4), squeeze=False)
    metric_keys = ("AP", "AP50", "AP75")
    colors_fp32 = "#4C72B0"
    colors_int8 = "#DD8452"

    for col, split in enumerate(splits):
        ax = axes[0, col]
        x = np.arange(len(metric_keys))
        w = 0.35
        fp32_vals = [data[split]["fp32"]["metrics"][k] for k in metric_keys]
        int8_vals = [data[split]["int8"]["metrics"][k] for k in metric_keys]
        ax.bar(x - w / 2, fp32_vals, w, label="fp32", color=colors_fp32)
        ax.bar(x + w / 2, int8_vals, w, label="int8", color=colors_int8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_keys)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"split: {split}")
        ax.set_ylabel("score")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("ONNX metrics by split (Google Drive data)", fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(report_dir, "metrics_by_split.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    saved.append(p1)

    # 2) Per-class AP per split
    for split in splits:
        classes = sorted(data[split]["fp32"]["per_class_ap"].keys())
        if not classes:
            continue
        fp32_c = [data[split]["fp32"]["per_class_ap"][c] for c in classes]
        int8_c = [data[split]["int8"]["per_class_ap"][c] for c in classes]

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.9), 4))
        x = np.arange(len(classes))
        w = 0.35
        ax.bar(x - w / 2, fp32_c, w, label="fp32", color=colors_fp32)
        ax.bar(x + w / 2, int8_c, w, label="int8", color=colors_int8)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AP per class")
        ax.set_title(f"Per-class AP — split: {split}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(report_dir, f"per_class_ap_{split}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    # 3) mAP delta int8 - fp32 per split
    fig, ax = plt.subplots(figsize=(5, 4))
    deltas = [
        data[s]["int8"]["mAP"] - data[s]["fp32"]["mAP"] for s in splits
    ]
    bars = ax.bar(splits, deltas, color=["#55A868" if d >= 0 else "#C44E52" for d in deltas])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("int8 mAP − fp32 mAP")
    ax.set_title("Quantization impact by split")
    ax.grid(axis="y", alpha=0.3)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{d:+.4f}",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=9)
    fig.tight_layout()
    p3 = os.path.join(report_dir, "map_delta_int8_vs_fp32.png")
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    saved.append(p3)

    print("Plots saved:")
    for p in saved:
        print(f"  {p}")
    return saved


def display_plots_in_notebook(plot_paths: list[str]) -> None:
    """Inline plot display for Colab/Jupyter."""
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    for path in plot_paths:
        if os.path.isfile(path):
            display(Image(filename=path))


def run_phase4_report(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    raw_dir: str,
    splits: list[str] | None = None,
    report_dir: str | None = None,
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
    show_plots: bool = True,
) -> dict[str, Any]:
    """
    Full Phase 4 report: mAP + extra COCO metrics + plots + JSON.

    raw_dir: Google Drive path in Colab, e.g. {DATA_DIR}/raw
    splits: default ['valid', 'test']
    report_dir: save artifacts (use Drive CHECKPOINT_DIR/eval_reports in Colab)
    """
    splits = list(splits or DEFAULT_SPLITS)
    assert_onnx_artifacts(fp32_onnx_path, int8_onnx_path)

    for split in splits:
        ann = resolve_ann_path(raw_dir, split)
        img_dir = resolve_split_dir(raw_dir, split)
        if not os.path.isfile(ann):
            raise FileNotFoundError(f"Split '{split}': missing {ann}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Split '{split}': missing images at {img_dir}")

    print(f"Phase 4 report")
    print(f"  raw_dir (Drive): {raw_dir}")
    print(f"  splits: {splits}")
    if max_samples is not None:
        print(f"  max_samples per split: {max_samples}")

    fp32_session = create_session(fp32_onnx_path)
    int8_session = create_session(int8_onnx_path)

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for split in splits:
        results[split] = {}
        print(f"\n========== split: {split} ==========")
        for tag, session in (("fp32", fp32_session), ("int8", int8_session)):
            results[split][tag] = evaluate_split(
                session,
                raw_dir,
                split,
                model_tag=tag,
                img_size=img_size,
                score_threshold=score_threshold,
                max_samples=max_samples,
            )

    _print_summary_table(results, splits)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": raw_dir,
        "splits": splits,
        "fp32_onnx": fp32_onnx_path,
        "int8_onnx": int8_onnx_path,
        "score_threshold": score_threshold,
        "max_samples": max_samples,
        "results": results,
    }

    plot_paths: list[str] = []
    if report_dir:
        save_report_json(report, report_dir)
        plot_paths = plot_phase4_report(report, report_dir)
        if show_plots:
            display_plots_in_notebook(plot_paths)

    report["plot_paths"] = plot_paths
    report["report_dir"] = report_dir
    return report


# Back-compat alias
def run_phase4_evaluation(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    raw_dir: str = "data/raw",
    splits: list[str] | None = None,
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
    report_dir: str | None = None,
) -> dict[str, Any]:
    return run_phase4_report(
        fp32_onnx_path=fp32_onnx_path,
        int8_onnx_path=int8_onnx_path,
        raw_dir=raw_dir,
        splits=splits,
        report_dir=report_dir,
        img_size=img_size,
        score_threshold=score_threshold,
        max_samples=max_samples,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 ONNX report (mAP + plots)")
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--report-dir", default="reports/phase4_eval")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_phase4_report(
        fp32_onnx_path=args.fp32,
        int8_onnx_path=args.int8,
        raw_dir=args.raw_dir,
        splits=args.splits,
        report_dir=args.report_dir,
        img_size=args.img_size,
        score_threshold=args.threshold,
        max_samples=args.max_samples,
        show_plots=not args.no_plots,
    )
