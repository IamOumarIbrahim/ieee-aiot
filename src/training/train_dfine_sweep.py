"""
train_dfine_sweep.py - Automated D-FINE-N Ratio Sweep Launcher & Evaluator.

Orchestrates sequential training across the 5 negative-ratio configurations using the upstream D-FINE engine:
- Model: D-FINE-N (HGNetv2-B0 backbone)
- 5 arithmetic ratio splits: 0%, 20%, 40%, 60%, 80%
- Frozen RTX 4060 protocol: physical batch 8, accum steps 4 (effective batch 32), 160 epochs, stop_epoch 148, seed 42
- Native COCO evaluation & FP/1k evaluation on negative benchmark test frames
- Generates runs/dfine_ratio_sweep/dfine_sweep_summary.json and .md matching YOLO sweep schema
"""

import os
import sys
import gc
import json
import time
import argparse
import subprocess
from pathlib import Path

# Add src/training to sys.path for dfine_utils
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src" / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "training"))

from dfine_utils import (
    find_dfine_checkpoint,
    build_dfine_train_cmd,
    load_dfine_model,
    evaluate_dfine_coco,
    calculate_dfine_fp_per_1k,
    patch_dfine_if_needed,
    DEFAULT_DFINE_AMP,
)

DFINE_SPLITS = [
    {"name": "dfine_00_pos_only", "ratio": "0%", "config": "configs/dfine/dfine_00_pos_only.yml"},
    {"name": "dfine_20_low_neg", "ratio": "20%", "config": "configs/dfine/dfine_20_low_neg.yml"},
    {"name": "dfine_40_mod_neg", "ratio": "40%", "config": "configs/dfine/dfine_40_mod_neg.yml"},
    {"name": "dfine_60_high_neg", "ratio": "60%", "config": "configs/dfine/dfine_60_high_neg.yml"},
    {"name": "dfine_80_max_neg", "ratio": "80%", "config": "configs/dfine/dfine_80_max_neg.yml"},
]

def parse_args():
    parser = argparse.ArgumentParser(description="Automated D-FINE Ratio Sweep Launcher & Evaluator")
    parser.add_argument("--dfine-dir", type=str, default="DFINE", help="Path to cloned upstream Peterande/D-FINE repository")
    parser.add_argument("--splits", type=str, default="00,20,40,60", help="Comma-separated split keys (e.g., '00,20,40,60' or 'all')")
    parser.add_argument("--device", type=str, default="0", help="CUDA visible device index")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable AMP for D-FINE training (default: False for Windows cuBLAS stability)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation on existing checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration paths and D-FINE entrypoint")
    return parser.parse_args()

def main():
    args = parse_args()
    repo_root = REPO_ROOT
    dfine_path = (repo_root / args.dfine_dir).resolve() if not Path(args.dfine_dir).is_absolute() else Path(args.dfine_dir).resolve()
    train_py = dfine_path / "train.py"

    selected_splits = DFINE_SPLITS
    if args.splits != "all":
        filter_keys = [k.strip() for k in args.splits.split(",")]
        selected_splits = [s for s in DFINE_SPLITS if any(k in s["name"] for k in filter_keys)]

    use_amp = args.amp
    project_dir = repo_root / "runs" / "dfine_ratio_sweep"
    os.makedirs(project_dir, exist_ok=True)

    print("============================================================")
    print("  D-FINE-N Negative-Frame Ratio Sweep Launcher & Evaluator")
    print(f"  Upstream D-FINE directory: {dfine_path}")
    print(f"  Target splits: {[s['name'] for s in selected_splits]}")
    print(f"  Protocol: batch=8, accum=4 (eff=32), epochs=160, stop_epoch=148, seed=42, amp={use_amp}")
    print(f"  Output Directory: {project_dir}")
    print("============================================================\n")

    test_manifest = repo_root / "data" / "processed" / "RGB" / "yolo" / "test.txt"
    instances_test = repo_root / "data" / "processed" / "RGB" / "coco" / "dfine" / "instances_test.json"

    if args.dry_run:
        print("[DRY-RUN] Checking D-FINE installation, configs, and test manifests:")
        print(f"  - D-FINE engine train.py exists: {train_py.exists()} ({train_py})")
        print(f"  - test_manifest exists: {test_manifest.exists()} ({test_manifest})")
        print(f"  - instances_test exists: {instances_test.exists()} ({instances_test})")

        from dfine_utils import find_dfine_checkpoint
        for s in selected_splits:
            cfg = repo_root / s["config"]
            exists = cfg.exists()
            status = "FOUND" if exists else "MISSING"
            ckpt = find_dfine_checkpoint(project_dir / s["name"])
            print(f"  - {s['name']}: [{status}] ({cfg}) | Existing checkpoint: {ckpt}")
        print("\nDry run completed successfully.")
        return

    if not train_py.exists():
        print(f"[ERROR] D-FINE train.py not found at: {train_py}")
        print("Please ensure Peterande/D-FINE exists in repo or specify --dfine-dir.")
        sys.exit(1)

    # Ensure Windows compatibility patches are applied to D-FINE
    patch_dfine_if_needed(dfine_path)

    # Ensure D-FINE can find data directory when executed from dfine_path
    dfine_data = dfine_path / "data"
    repo_data = repo_root / "data"
    if not dfine_data.exists() and repo_data.exists():
        try:
            if sys.platform == "win32":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(dfine_data), str(repo_data)], check=False, stdout=subprocess.DEVNULL)
            else:
                os.symlink(str(repo_data), str(dfine_data))
        except Exception:
            pass

    summary_file = project_dir / "dfine_sweep_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = None
    else:
        summary = None

    if not summary or "runs" not in summary:
        summary = {
            "model": "dfine",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": {
                "epochs": 160,
                "batch": 8,
                "accum_steps": 4,
                "effective_batch": 32,
                "stop_epoch": 148,
                "seed": 42,
                "amp": use_amp,
            },
            "runs": []
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device

    for idx, s in enumerate(selected_splits, 1):
        split_name = s["name"]
        ratio_label = s["ratio"]
        cfg_abs = (repo_root / s["config"]).resolve()
        run_output_dir = project_dir / split_name
        os.makedirs(run_output_dir, exist_ok=True)

        print(f"\n[{idx}/{len(selected_splits)}] Target: {split_name} ({ratio_label} Negatives)")
        start_time = time.time()

        if not args.eval_only:
            cmd = build_dfine_train_cmd(
                dfine_train_py=train_py,
                config_path=cfg_abs,
                use_amp=use_amp,
                seed=42,
                output_dir=run_output_dir,
                device=args.device
            )

            print(f"  Command: {' '.join(cmd)}")
            ret = subprocess.run(cmd, cwd=str(dfine_path), env=env)
            if ret.returncode != 0:
                print(f"[ERROR] Run {split_name} failed with returncode {ret.returncode}")
                sys.exit(ret.returncode)
            else:
                print(f"  [OK] Training completed for {split_name}.")

        elapsed_min = (time.time() - start_time) / 60.0

        # Discover best/latest checkpoint
        ckpt_path = find_dfine_checkpoint(run_output_dir)
        if ckpt_path is None:
            print(f"[ERROR] No checkpoint found for {split_name} in {run_output_dir}")
            sys.exit(1)

        print(f"  Evaluating {split_name} checkpoint: {ckpt_path.name} on test set...")
        map50, map50_95, precision, recall = evaluate_dfine_coco(
            config_path=cfg_abs,
            checkpoint_path=ckpt_path,
            test_ann_file=instances_test,
            device=args.device,
            dfine_dir=str(dfine_path)
        )

        print(f"  Calculating false-positive rate on negative test frames (conf=0.25)...")
        model_wrapper, _ = load_dfine_model(
            config_path=cfg_abs,
            checkpoint_path=ckpt_path,
            device=args.device,
            deploy=True,
            dfine_dir=str(dfine_path)
        )

        fp_per_1k, total_fp, test_negs = calculate_dfine_fp_per_1k(
            model_or_wrapper=model_wrapper,
            test_manifest_path=test_manifest,
            conf_thresh=0.25,
            device=args.device
        )

        # Free model from GPU memory
        del model_wrapper
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        # Update or append run result
        summary["runs"] = [r for r in summary["runs"] if r["split"] != split_name]
        summary["runs"].append(run_result)
        # Sort runs by split order
        order = {s["name"]: i for i, s in enumerate(DFINE_SPLITS)}
        summary["runs"].sort(key=lambda r: order.get(r["split"], 99))

        print(f"  Completed {split_name} -> mAP@50: {map50:.4f}, mAP@50:95: {map50_95:.4f}, FP/1k: {fp_per_1k:.2f}")

        # Save interim summary JSON and Markdown after each evaluation
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        md_file = project_dir / "dfine_sweep_summary.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Sweep Summary: D-FINE-N\n\n")
            f.write(f"- **Timestamp:** {summary['timestamp']}\n")
            f.write(f"- **Protocol:** epochs={summary['protocol']['epochs']}, batch={summary['protocol']['batch']}, accum_steps={summary['protocol']['accum_steps']}, stop_epoch={summary['protocol']['stop_epoch']}, seed={summary['protocol']['seed']}, amp={summary['protocol']['amp']}\n\n")
            f.write("| Split | Ratio | mAP@50 | mAP@50:95 | Precision | Recall | FP/1k | Train Time (min) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in summary["runs"]:
                f.write(f"| `{r['split']}` | {r['ratio']} | {r['test_map50']:.4f} | {r['test_map50_95']:.4f} | {r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_fp_per_1k']:.2f} | {r['train_time_minutes']:.1f} |\n")

    # Print final summary table
    print("\n" + "="*85)
    print("  D-FINE-N RATIO SWEEP SUMMARY TABLE")
    print("="*85)
    print(f"{'Split':<22} | {'Ratio':<6} | {'mAP@50':<8} | {'mAP@50:95':<10} | {'Precision':<10} | {'Recall':<8} | {'FP/1k':<8} | {'Time (min)':<10}")
    print("-" * 95)
    for r in summary["runs"]:
        print(f"{r['split']:<22} | {r['ratio']:<6} | {r['test_map50']:<8.4f} | {r['test_map50_95']:<10.4f} | {r['test_precision']:<10.4f} | {r['test_recall']:<8.4f} | {r['test_fp_per_1k']:<8.2f} | {r['train_time_minutes']:<10.1f}")
    print("="*85)
    print(f"Summary persisted to:\n  - {summary_file}\n  - {project_dir / 'dfine_sweep_summary.md'}\n")

if __name__ == "__main__":
    main()
