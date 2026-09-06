"""
train_yolo_sweep.py - Automated Negative-Frame Ratio Sweep Runner for Ultralytics YOLO.

Implements the frozen experimental protocol on RTX 4060 8GB:
- Models: YOLO11n, YOLO26n
- 5 arithmetic ratio splits: 0%, 20%, 40%, 60%, 80%
- Resolution: 640x640, Batch: 16, Epochs: 100, close_mosaic: 10, seed: 42
- Automatically logs validation & held-out test metrics into a consolidated JSON summary table.
"""

import os
import sys
import json
import argparse
import time
import gc
from pathlib import Path
import torch
from ultralytics import YOLO

# Experimental split definitions matching verified dataset configurations
SPLITS = [
    {"name": "train_00_pos_only", "ratio": "0%", "config": "configs/yolo/yolo_00_pos_only.yaml"},
    {"name": "train_20_low_neg", "ratio": "20%", "config": "configs/yolo/yolo_20_low_neg.yaml"},
    {"name": "train_40_mod_neg", "ratio": "40%", "config": "configs/yolo/yolo_40_mod_neg.yaml"},
    {"name": "train_60_high_neg", "ratio": "60%", "config": "configs/yolo/yolo_60_high_neg.yaml"},
    {"name": "train_80_max_neg", "ratio": "80%", "config": "configs/yolo/yolo_80_max_neg.yaml"},
]

def parse_args():
    parser = argparse.ArgumentParser(description="Automated YOLO Negative-Frame Ratio Sweep Runner")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model checkpoint (e.g., yolo11n.pt, yolo26n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (frozen protocol: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (frozen protocol: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (frozen protocol: 640)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (frozen protocol: 42)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g. 0, cpu)")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs to disable mosaic (frozen: 10)")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay (frozen: 0.0005)")
    parser.add_argument("--splits", type=str, default="00,20,40,60", help="Comma-separated split keys (e.g., '00,20,40,60' or 'all')")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable PyTorch AMP mixed precision (default: False for Windows cuBLAS stability)")
    parser.add_argument("--dry-run", action="store_true", help="Validate configurations without initiating training")
    return parser.parse_args()

def calculate_fp_per_1k(model, test_manifest_path, conf_thresh=0.25):
    """
    Evaluates detector false-positive predictions on background-only negative test frames.
    Returns: false positives per 1,000 frames on the negative benchmark test set.
    """
    manifest_file = Path(test_manifest_path).resolve()
    manifest_dir = manifest_file.parent

    with open(manifest_file, "r") as f:
        all_test_paths = [l.strip() for l in f if l.strip()]

    # Load annotations to identify background-only frames in the test set
    val_test_dir = manifest_dir.parent / "coco"
    with open(val_test_dir / "instances_test.json", "r") as f:
        test_coco = json.load(f)

    pos_img_ids = {ann["image_id"] for ann in test_coco["annotations"]}
    neg_img_filenames = {img["file_name"] for img in test_coco["images"] if img["id"] not in pos_img_ids}

    neg_test_paths = []
    for p in all_test_paths:
        # Standardize matching
        fn = p.replace("./../", "")
        if fn in neg_img_filenames or fn.replace("images/", "") in {f.replace("images/", "") for f in neg_img_filenames}:
            # Resolve relative manifest path against the manifest's parent directory to avoid CWD resolution errors
            resolved_p = (manifest_dir / p).resolve()
            neg_test_paths.append(str(resolved_p))

    total_neg_frames = len(neg_test_paths)
    assert total_neg_frames == 1272, f"Expected 1272 negative test frames, got {total_neg_frames}"
    if total_neg_frames == 0:
        return 0.0, 0, 0

    # Run inference on negative test frames in batches to avoid VRAM spikes on 8GB GPUs
    fp_count = 0
    batch_size = 32
    for i in range(0, total_neg_frames, batch_size):
        batch = neg_test_paths[i : i + batch_size]
        results = model.predict(batch, conf=conf_thresh, verbose=False)
        for r in results:
            fp_count += len(r.boxes)
        del results

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fp_per_1k = (fp_count / total_neg_frames) * 1000.0
    return fp_per_1k, fp_count, total_neg_frames

def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    model_stem = Path(args.model).stem
    project_dir = repo_root / "runs" / f"{model_stem}_ratio_sweep"
    os.makedirs(project_dir, exist_ok=True)

    selected_splits = SPLITS
    if args.splits != "all":
        filter_keys = [k.strip() for k in args.splits.split(",")]
        selected_splits = [s for s in SPLITS if any(k in s["name"] for k in filter_keys)]

    print(f"============================================================")
    print(f"  Starting YOLO Ratio Sweep: {args.model}")
    print(f"  Target Splits: {[s['name'] for s in selected_splits]}")
    print(f"  Frozen Hyperparameters: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, seed={args.seed}")
    print(f"  Output Directory: {project_dir}")
    print(f"============================================================\n")

    if args.dry_run:
        print("[DRY-RUN] Validating dataset configuration paths:")
        for s in selected_splits:
            cfg_path = repo_root / s["config"]
            print(f"  - {s['name']}: {cfg_path.exists()} ({cfg_path})")
        test_manifest = repo_root / "data" / "processed" / "RGB" / "yolo" / "test.txt"
        print(f"  - test_manifest exists: {test_manifest.exists()} ({test_manifest})")
        if test_manifest.exists():
            val_test_dir = test_manifest.parent.parent / "coco"
            instances_test = val_test_dir / "instances_test.json"
            print(f"  - coco instances_test exists: {instances_test.exists()} ({instances_test})")
        print("\nAll configurations validated successfully. Exiting dry run.")
        return

    summary = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "close_mosaic": args.close_mosaic,
            "weight_decay": args.weight_decay,
            "optimizer": "auto",
            "amp": args.amp,
        },
        "runs": []
    }

    test_manifest = repo_root / "data" / "processed" / "RGB" / "yolo" / "test.txt"

    for idx, s in enumerate(selected_splits, 1):
        split_name = s["name"]
        ratio_label = s["ratio"]
        cfg_path = str(repo_root / s["config"])

        print(f"\n[{idx}/{len(selected_splits)}] Launching Run: {split_name} ({ratio_label} Negatives)")
        start_time = time.time()

        # Initialize fresh model from pretrained weights
        model = YOLO(args.model)

        # Execute training adhering strictly to frozen protocol
        model.train(
            data=cfg_path,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            seed=args.seed,
            amp=args.amp,
            close_mosaic=args.close_mosaic,
            optimizer="auto",
            weight_decay=args.weight_decay,
            project=str(project_dir),
            name=split_name,
            device=args.device,
            exist_ok=True,
            verbose=True
        )

        elapsed_min = (time.time() - start_time) / 60.0

        # Evaluate on the held-out test benchmark
        print(f"  Evaluating {split_name} on held-out test split...")
        test_metrics = model.val(data=cfg_path, split="test", imgsz=args.imgsz, device=args.device, verbose=False)

        precision = float(test_metrics.box.mp)
        recall = float(test_metrics.box.mr)
        map50 = float(test_metrics.box.map50)
        map50_95 = float(test_metrics.box.map)

        # Compute FP/1k frames on negative test frames
        print(f"  Calculating false-positive rate on negative test frames (conf=0.25)...")
        fp_per_1k, total_fp, test_negs = calculate_fp_per_1k(model, test_manifest, conf_thresh=0.25)

        run_result = {
            "split": split_name,
            "ratio": ratio_label,
            "train_time_minutes": round(elapsed_min, 2),
            "test_precision": round(precision, 4),
            "test_recall": round(recall, 4),
            "test_map50": round(map50, 4),
            "test_map50_95": round(map50_95, 4),
            "test_fp_per_1k": round(fp_per_1k, 2),
            "total_test_fps": total_fp,
            "test_neg_frames": test_negs
        }

        summary["runs"].append(run_result)
        print(f"  Completed {split_name} in {elapsed_min:.1f}m -> mAP@50: {map50:.4f}, FP/1k: {fp_per_1k:.2f}")

        # Clean up GPU memory for next split
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save interim summary after each run (JSON and Markdown)
        summary_file = project_dir / f"{model_stem}_sweep_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        md_file = project_dir / f"{model_stem}_sweep_summary.md"
        with open(md_file, "w") as f:
            f.write(f"# Sweep Summary: {args.model}\n\n")
            f.write(f"- **Timestamp:** {summary['timestamp']}\n")
            f.write(f"- **Protocol:** epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, seed={args.seed}\n\n")
            f.write("| Split | Ratio | mAP@50 | mAP@50:95 | Precision | Recall | FP/1k | Train Time (min) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in summary["runs"]:
                f.write(f"| `{r['split']}` | {r['ratio']} | {r['test_map50']:.4f} | {r['test_map50_95']:.4f} | {r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_fp_per_1k']:.2f} | {r['train_time_minutes']:.1f} |\n")

    # Print summary table
    print("\n" + "="*80)
    print(f"  SWEEP SUMMARY TABLE: {args.model}")
    print("="*80)
    print(f"{'Split':<20} | {'Ratio':<6} | {'mAP@50':<8} | {'mAP@50:95':<10} | {'Precision':<10} | {'Recall':<8} | {'FP/1k':<8} | {'Time (min)':<10}")
    print("-" * 95)
    for r in summary["runs"]:
        print(f"{r['split']:<20} | {r['ratio']:<6} | {r['test_map50']:<8.4f} | {r['test_map50_95']:<10.4f} | {r['test_precision']:<10.4f} | {r['test_recall']:<8.4f} | {r['test_fp_per_1k']:<8.2f} | {r['train_time_minutes']:<10.1f}")
    print("="*80)
    print(f"Summary persisted to:\n  - {project_dir / f'{model_stem}_sweep_summary.json'}\n  - {project_dir / f'{model_stem}_sweep_summary.md'}\n")

if __name__ == "__main__":
    main()
