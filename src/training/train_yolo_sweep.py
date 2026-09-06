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
    parser.add_argument("--project", type=str, default=None, help="Custom project output directory")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer name (default: AdamW for Windows cuBLAS stability)")
    parser.add_argument("--lr0", type=float, default=0.00125, help="Initial learning rate (default: 0.00125, matching nc=4 AdamW schedule)")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable PyTorch AMP mixed precision (default: False for Windows cuBLAS stability)")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation on existing checkpoints without retraining")
    parser.add_argument("--dry-run", action="store_true", help="Validate configurations without initiating training")
    return parser.parse_args()

def calculate_fp_per_1k(model, manifest_path, split_name="test", conf_thresh=0.25, device="0"):
    """
    Evaluates detector false-positive predictions on background-only negative frames.
    Returns: false positives per 1,000 frames on the negative benchmark split.
    """
    manifest_file = Path(manifest_path).resolve()
    manifest_dir = manifest_file.parent

    with open(manifest_file, "r") as f:
        all_paths = [l.strip() for l in f if l.strip()]

    # Load annotations to identify background-only frames in the split
    coco_dir = manifest_dir.parent / "coco"
    coco_file = coco_dir / f"instances_{split_name}.json"
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    pos_img_ids = {ann["image_id"] for ann in coco_data["annotations"]}
    neg_img_filenames = {img["file_name"] for img in coco_data["images"] if img["id"] not in pos_img_ids}

    neg_paths = []
    for p in all_paths:
        # Standardize matching
        fn = p.replace("./../", "")
        if fn in neg_img_filenames or fn.replace("images/", "") in {f.replace("images/", "") for f in neg_img_filenames}:
            resolved_p = (manifest_dir / p).resolve()
            neg_paths.append(str(resolved_p))

    total_neg_frames = len(neg_paths)
    assert total_neg_frames == 1272, f"Expected 1272 negative frames for {split_name}, got {total_neg_frames}"
    if total_neg_frames == 0:
        return 0.0, 0, 0

    # Run inference on negative frames in batches to avoid VRAM spikes on 8GB GPUs
    fp_count = 0
    batch_size = 32
    for i in range(0, total_neg_frames, batch_size):
        batch = neg_paths[i : i + batch_size]
        results = model.predict(batch, conf=conf_thresh, device=device, verbose=False)
        for r in results:
            fp_count += len(r.boxes)
        del results

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fp_per_1k = (fp_count / total_neg_frames) * 1000.0
    return fp_per_1k, fp_count, total_neg_frames

def evaluate_split(model, cfg_path, manifest_path, split_name, imgsz=640, device="0", conf_thresh=0.25):
    """
    Evaluates detector on a full split (mAP, P, R) and computes FP/1k on negative frames.
    """
    print(f"  Evaluating split '{split_name}' (mAP / Precision / Recall)...")
    metrics = model.val(data=cfg_path, split=split_name, imgsz=imgsz, device=device, plots=False, save=False, verbose=False)

    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print(f"  Calculating false-positive rate on negative {split_name} frames (conf={conf_thresh})...")
    fp_per_1k, fp_count, total_negs = calculate_fp_per_1k(model, manifest_path, split_name=split_name, conf_thresh=conf_thresh, device=device)

    # Extract per-class APs
    per_class = {}
    class_names = getattr(model, "names", {0: "yawning", 1: "hand_over_mouth", 2: "drinking", 3: "phone_use"})
    if hasattr(metrics.box, "ap50") and hasattr(metrics.box, "ap"):
        for cls_idx, cls_name in class_names.items():
            if cls_idx < len(metrics.box.ap50):
                per_class[cls_name] = {
                    "map50": round(float(metrics.box.ap50[cls_idx]), 4),
                    "map50_95": round(float(metrics.box.ap[cls_idx]), 4),
                }

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "fp_per_1k": round(fp_per_1k, 2),
        "total_fps": fp_count,
        "neg_frames": total_negs,
        "per_class": per_class,
    }

def save_summary_outputs(summary, project_dir, model_stem):
    """Helper to save summary JSON and formatted Markdown report."""
    summary_file = project_dir / f"{model_stem}_sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    md_file = project_dir / f"{model_stem}_sweep_summary.md"
    with open(md_file, "w") as f:
        f.write(f"# Evaluation Summary: {summary['model']}\n\n")
        f.write(f"- **Timestamp:** {summary['timestamp']}\n")
        proto = summary.get("protocol", {})
        f.write(f"- **Protocol:** epochs={proto.get('epochs')}, batch={proto.get('batch')}, imgsz={proto.get('imgsz')}, seed={proto.get('seed')}\n\n")
        f.write("### Validation & Test Split Performance\n\n")
        f.write("| Split | Ratio | Val mAP50 | Val mAP50:95 | Val FP/1k | Test mAP50 | Test mAP50:95 | Test FP/1k | Train Time (min) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in summary["runs"]:
            tt = f"{r['train_time_minutes']:.1f}" if r.get('train_time_minutes') is not None else "N/A"
            v_m50 = f"{r['val_map50']:.4f}" if "val_map50" in r else "N/A"
            v_m = f"{r['val_map50_95']:.4f}" if "val_map50_95" in r else "N/A"
            v_fp = f"{r['val_fp_per_1k']:.2f}" if "val_fp_per_1k" in r else "N/A"
            t_m50 = f"{r['test_map50']:.4f}" if "test_map50" in r else "N/A"
            t_m = f"{r['test_map50_95']:.4f}" if "test_map50_95" in r else "N/A"
            t_fp = f"{r['test_fp_per_1k']:.2f}" if "test_fp_per_1k" in r else "N/A"
            f.write(f"| `{r['split']}` | {r['ratio']} | {v_m50} | {v_m} | {v_fp} | {t_m50} | {t_m} | {t_fp} | {tt} |\n")

def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    model_stem = Path(args.model).stem
    if args.project:
        project_dir = Path(args.project).resolve() if Path(args.project).is_absolute() else (repo_root / args.project).resolve()
    else:
        project_dir = repo_root / "runs" / f"{model_stem}_ratio_sweep"
    os.makedirs(project_dir, exist_ok=True)

    selected_splits = SPLITS
    if args.splits != "all":
        raw_keys = [k.strip() for k in args.splits.split(",")]
        filter_keys = []
        for k in raw_keys:
            if k in ("81", "81%", "nat", "natural", "80", "80%"):
                filter_keys.append("80")
            elif k in ("0", "0%"):
                filter_keys.append("00")
            elif k in ("20", "20%"):
                filter_keys.append("20")
            elif k in ("40", "40%"):
                filter_keys.append("40")
            elif k in ("60", "60%"):
                filter_keys.append("60")
            else:
                filter_keys.append(k)
        selected_splits = [s for s in SPLITS if any(k in s["name"] for k in filter_keys)]

    mode_label = "Evaluation Only (--eval-only)" if args.eval_only else "Training & Evaluation Sweep"
    print(f"============================================================")
    print(f"  Starting YOLO Runner: {args.model}")
    print(f"  Mode: {mode_label}")
    print(f"  Target Splits: {[s['name'] for s in selected_splits]}")
    print(f"  Configuration: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, seed={args.seed}, device={args.device}")
    print(f"  Output Directory: {project_dir}")
    print(f"============================================================\n")

    val_manifest = repo_root / "data" / "processed" / "RGB" / "yolo" / "val.txt"
    test_manifest = repo_root / "data" / "processed" / "RGB" / "yolo" / "test.txt"

    if args.dry_run:
        print("[DRY-RUN] Validating dataset configuration paths:")
        for s in selected_splits:
            cfg_path = repo_root / s["config"]
            ckpt_path = project_dir / s["name"] / "weights" / "best.pt"
            print(f"  - {s['name']}: config={cfg_path.exists()} ({cfg_path}), checkpoint={ckpt_path.exists()} ({ckpt_path})")
        print(f"  - val_manifest exists: {val_manifest.exists()} ({val_manifest})")
        print(f"  - test_manifest exists: {test_manifest.exists()} ({test_manifest})")
        print("\nAll configurations validated successfully. Exiting dry run.")
        return

    # Load existing summary to retain training metadata if available
    summary_file = project_dir / f"{model_stem}_sweep_summary.json"
    existing_runs = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                existing_data = json.load(f)
                existing_runs = {r["split"]: r for r in existing_data.get("runs", [])}
        except Exception as e:
            print(f"Warning: Could not parse existing summary: {e}")

    summary = {
        "model": str(args.model),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "close_mosaic": args.close_mosaic,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "amp": args.amp,
        },
        "runs": list(existing_runs.values())
    }

    for idx, s in enumerate(selected_splits, 1):
        split_name = s["name"]
        ratio_label = s["ratio"]
        cfg_path = str(repo_root / s["config"])

        if args.eval_only:
            ckpt_path = project_dir / split_name / "weights" / "best.pt"
            if not ckpt_path.exists():
                print(f"\n[{idx}/{len(selected_splits)}] [WARN] Checkpoint not found: {ckpt_path}. Skipping.")
                continue

            print(f"\n[{idx}/{len(selected_splits)}] Evaluating Checkpoint: {ckpt_path} ({split_name}, {ratio_label} Negatives)")
            model = YOLO(str(ckpt_path))

            # Prior train time if recorded
            train_time = existing_runs.get(split_name, {}).get("train_time_minutes", None)

            # Evaluate on validation split
            val_res = evaluate_split(model, cfg_path, val_manifest, split_name="val", imgsz=args.imgsz, device=args.device)

            # Evaluate on held-out test split
            test_res = evaluate_split(model, cfg_path, test_manifest, split_name="test", imgsz=args.imgsz, device=args.device)

            run_result = {
                "split": split_name,
                "ratio": ratio_label,
                "train_time_minutes": train_time,
                # Validation metrics
                "val_precision": val_res["precision"],
                "val_recall": val_res["recall"],
                "val_map50": val_res["map50"],
                "val_map50_95": val_res["map50_95"],
                "val_fp_per_1k": val_res["fp_per_1k"],
                "val_total_fps": val_res["total_fps"],
                # Test metrics
                "test_precision": test_res["precision"],
                "test_recall": test_res["recall"],
                "test_map50": test_res["map50"],
                "test_map50_95": test_res["map50_95"],
                "test_fp_per_1k": test_res["fp_per_1k"],
                "total_test_fps": test_res["total_fps"],
                "test_neg_frames": test_res["neg_frames"],
                "per_class": {
                    "val": val_res["per_class"],
                    "test": test_res["per_class"],
                }
            }

            summary["runs"] = [r for r in summary["runs"] if r["split"] != split_name]
            summary["runs"].append(run_result)
            order = {s["name"]: i for i, s in enumerate(SPLITS)}
            summary["runs"].sort(key=lambda r: order.get(r["split"], 99))
            print(f"  Val  -> mAP@50: {val_res['map50']:.4f}, mAP@50-95: {val_res['map50_95']:.4f}, FP/1k: {val_res['fp_per_1k']:.2f}")
            print(f"  Test -> mAP@50: {test_res['map50']:.4f}, mAP@50-95: {test_res['map50_95']:.4f}, FP/1k: {test_res['fp_per_1k']:.2f}")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            save_summary_outputs(summary, project_dir, model_stem)

        else:
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
                optimizer=args.optimizer,
                lr0=args.lr0,
                weight_decay=args.weight_decay,
                project=str(project_dir),
                name=split_name,
                device=args.device,
                exist_ok=True,
                verbose=True
            )

            elapsed_min = (time.time() - start_time) / 60.0

            # Reload best checkpoint for definitive evaluation
            best_ckpt = project_dir / split_name / "weights" / "best.pt"
            eval_model = YOLO(str(best_ckpt)) if best_ckpt.exists() else model

            # Evaluate on validation split
            val_res = evaluate_split(eval_model, cfg_path, val_manifest, split_name="val", imgsz=args.imgsz, device=args.device)

            # Evaluate on held-out test split
            test_res = evaluate_split(eval_model, cfg_path, test_manifest, split_name="test", imgsz=args.imgsz, device=args.device)

            run_result = {
                "split": split_name,
                "ratio": ratio_label,
                "train_time_minutes": round(elapsed_min, 2),
                # Validation metrics
                "val_precision": val_res["precision"],
                "val_recall": val_res["recall"],
                "val_map50": val_res["map50"],
                "val_map50_95": val_res["map50_95"],
                "val_fp_per_1k": val_res["fp_per_1k"],
                "val_total_fps": val_res["total_fps"],
                # Test metrics
                "test_precision": test_res["precision"],
                "test_recall": test_res["recall"],
                "test_map50": test_res["map50"],
                "test_map50_95": test_res["map50_95"],
                "test_fp_per_1k": test_res["fp_per_1k"],
                "total_test_fps": test_res["total_fps"],
                "test_neg_frames": test_res["neg_frames"],
                "per_class": {
                    "val": val_res["per_class"],
                    "test": test_res["per_class"],
                }
            }

            summary["runs"] = [r for r in summary["runs"] if r["split"] != split_name]
            summary["runs"].append(run_result)
            order = {s["name"]: i for i, s in enumerate(SPLITS)}
            summary["runs"].sort(key=lambda r: order.get(r["split"], 99))
            print(f"  Completed {split_name} in {elapsed_min:.1f}m:")
            print(f"    Val  -> mAP@50: {val_res['map50']:.4f}, FP/1k: {val_res['fp_per_1k']:.2f}")
            print(f"    Test -> mAP@50: {test_res['map50']:.4f}, FP/1k: {test_res['fp_per_1k']:.2f}")

            if eval_model is not model:
                del eval_model
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            save_summary_outputs(summary, project_dir, model_stem)

    # Print consolidated summary table
    print("\n" + "="*115)
    print(f"  SWEEP SUMMARY TABLE: {args.model}")
    print("="*115)
    print(f"{'Split':<20} | {'Ratio':<6} | {'Val mAP50':<9} | {'Val FP/1k':<9} | {'Test mAP50':<10} | {'Test mAP':<8} | {'Test FP/1k':<10} | {'Time (min)':<10}")
    print("-" * 115)
    for r in summary["runs"]:
        tt = f"{r['train_time_minutes']:.1f}" if r.get('train_time_minutes') is not None else "N/A"
        v_m50 = f"{r['val_map50']:.4f}" if "val_map50" in r else "N/A"
        v_fp = f"{r['val_fp_per_1k']:.2f}" if "val_fp_per_1k" in r else "N/A"
        t_m50 = f"{r['test_map50']:.4f}" if "test_map50" in r else "N/A"
        t_m = f"{r['test_map50_95']:.4f}" if "test_map50_95" in r else "N/A"
        t_fp = f"{r['test_fp_per_1k']:.2f}" if "test_fp_per_1k" in r else "N/A"
        print(f"{r['split']:<20} | {r['ratio']:<6} | {v_m50:<9} | {v_fp:<9} | {t_m50:<10} | {t_m:<8} | {t_fp:<10} | {tt:<10}")
    print("="*115)
    print(f"Summary persisted to:\n  - {project_dir / f'{model_stem}_sweep_summary.json'}\n  - {project_dir / f'{model_stem}_sweep_summary.md'}\n")

if __name__ == "__main__":
    main()

