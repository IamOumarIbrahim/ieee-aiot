"""
run_rq2_curated_sweep.py - Automated RQ2 Curated Hard-Negative Training & Evaluation Sweep.

Protocol:
- Models: YOLO11n, YOLO26n, YOLO12n
- Baseline checkpoints: runs/{model}_ratio_sweep/train_00_pos_only/weights/best.pt
- Curated datasets: configs/yolo/yolo_curated_{model}_best_curated.yaml (N=4,001 frames: 2,401 pos, 1,600 neg)
- Training: 100 epochs, batch 16, imgsz 640, seed 42, amp=False (FP32), close_mosaic=10, optimizer=AdamW, lr0=0.00125, weight_decay=0.0005
- Evaluation: Precision, Recall, mAP50, mAP50-95, FP/1k on test negative frames at tau=0.25
- Output: runs/rq2_curation_summary.json, Table IV in main.tex, recompile main.pdf
"""

import os
import sys
import json
import time
import re
import argparse
import subprocess
import gc
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PDFLATEX_PATH = Path(r"C:\Users\omarb\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")

if str(REPO_ROOT / "src" / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "training"))

from train_yolo_sweep import calculate_fp_per_1k, evaluate_split

MODELS = [
    {"key": "yolo11n", "base_weights": "yolo11n.pt"},
    {"key": "yolo26n", "base_weights": "yolo26n.pt"},
    {"key": "yolo12n", "base_weights": "yolo12n.pt"},
    {"key": "yolov10n", "base_weights": "yolov10n.pt"},
]

def parse_args():
    parser = argparse.ArgumentParser(description="RQ2 Hard-Negative Curated Training & Evaluation Runner")
    parser.add_argument("--models", type=str, default="yolo11n,yolo26n,yolo12n,yolov10n", help="Comma-separated model keys to run")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate existing checkpoints")
    parser.add_argument("--populate-only", action="store_true", help="Only populate tables in main.tex from existing summary")
    return parser.parse_args()

def load_rq1_random_result(model_key, seed=42):
    sweep_dir = REPO_ROOT / "runs" / (f"{model_key}_ratio_sweep" if seed == 42 else f"{model_key}_ratio_sweep_seed{seed}")
    sweep_file = sweep_dir / f"{model_key}_sweep_summary.json"
    if not sweep_file.exists():
        return None
    with open(sweep_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data.get("runs", []):
        if r.get("ratio") == "40%":
            return r
    return None

def train_curated_model(model_key, base_weights, args):
    tag = f"{model_key}_best_curated"
    cfg_path = REPO_ROOT / "configs" / "yolo" / f"yolo_curated_{tag}.yaml"
    project_dir = REPO_ROOT / "runs" / (f"{model_key}_ratio_sweep" if args.seed == 42 else f"{model_key}_ratio_sweep_seed{args.seed}")
    run_name = f"train_curated_{tag}"
    weights_dir = project_dir / run_name / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Curated dataset config not found: {cfg_path}")

    # Check if already completed
    if best_ckpt.exists() and not args.eval_only:
        try:
            ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            ep = ckpt_data.get("epoch", -1)
            print(f"[{model_key.upper()}] Existing checkpoint found: {best_ckpt} (epoch {ep})")
            if ep >= args.epochs - 1:
                print(f"[{model_key.upper()}] Training already completed {args.epochs} epochs. Skipping training.")
                return best_ckpt, None
        except Exception as e:
            print(f"[{model_key.upper()}] Could not inspect checkpoint: {e}")

    if args.eval_only:
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found for eval-only: {best_ckpt}")
        return best_ckpt, None

    print("\n" + "="*70)
    print(f"  LAUNCHING RQ2 CURATED TRAINING: {model_key.upper()}")
    print(f"  Config: {cfg_path}")
    print(f"  Base Weights: {base_weights}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch}, Imgsz: {args.imgsz}, Device: {args.device}")
    print("="*70 + "\n")

    start_time = time.time()

    from ultralytics import YOLO

    resume = False
    if last_ckpt.exists():
        try:
            last_data = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            last_ep = last_data.get("epoch", -1)
            if 0 <= last_ep < args.epochs - 1:
                print(f"[{model_key.upper()}] Resuming interrupted training from epoch {last_ep + 1}...")
                resume = True
        except Exception:
            pass

    if resume:
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        model = YOLO(str(REPO_ROOT / base_weights))
        model.train(
            data=str(cfg_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            seed=args.seed,
            amp=False,
            close_mosaic=10,
            optimizer="AdamW",
            lr0=0.00125,
            weight_decay=0.0005,
            project=str(project_dir),
            name=run_name,
            device=args.device,
            exist_ok=True,
            verbose=True,
        )

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"\n[{model_key.upper()}] Training finished in {elapsed_min:.2f} minutes.")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_ckpt, round(elapsed_min, 2)

def evaluate_curated_model(model_key, ckpt_path, train_time, args):
    tag = f"{model_key}_best_curated"
    cfg_path = str(REPO_ROOT / "configs" / "yolo" / f"yolo_curated_{tag}.yaml")
    val_manifest = REPO_ROOT / "data" / "processed" / "RGB" / "yolo" / "val.txt"
    test_manifest = REPO_ROOT / "data" / "processed" / "RGB" / "yolo" / "test.txt"

    from ultralytics import YOLO

    print(f"\n[{model_key.upper()}] Evaluating best checkpoint: {ckpt_path}")
    eval_model = YOLO(str(ckpt_path))

    val_res = evaluate_split(eval_model, cfg_path, val_manifest, split_name="val", imgsz=args.imgsz, device=args.device)
    test_res = evaluate_split(eval_model, cfg_path, test_manifest, split_name="test", imgsz=args.imgsz, device=args.device)

    print(f"  [{model_key.upper()}] Validation -> mAP@50: {val_res['map50']:.4f}, mAP@50:95: {val_res['map50_95']:.4f}, FP/1k: {val_res['fp_per_1k']:.2f}")
    print(f"  [{model_key.upper()}] Held-Out Test -> mAP@50: {test_res['map50']:.4f}, mAP@50:95: {test_res['map50_95']:.4f}, FP/1k: {test_res['fp_per_1k']:.2f}")

    del eval_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train_time_minutes": train_time,
        "val_precision": val_res["precision"],
        "val_recall": val_res["recall"],
        "val_map50": val_res["map50"],
        "val_map50_95": val_res["map50_95"],
        "val_fp_per_1k": val_res["fp_per_1k"],
        "val_total_fps": val_res["total_fps"],
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

def update_manuscript_table_iv(rq2_summary):
    tex_path = REPO_ROOT / "docs" / "manuscript" / "main.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    name_map = {
        "yolo11n": "YOLO11n",
        "yolo26n": "YOLO26n",
        "yolo12n": "YOLO12n",
        "yolov10n": "YOLOv10n",
    }

    for model_key, det_name in name_map.items():
        if model_key not in rq2_summary:
            continue
        entry = rq2_summary[model_key]
        rand = entry.get("random", {})
        cur = entry.get("curated", {})

        r_m50 = rand.get("test_map50")
        r_m95 = rand.get("test_map50_95")
        r_fp = rand.get("test_fp_per_1k")

        c_m50 = cur.get("test_map50")
        c_m95 = cur.get("test_map50_95")
        c_fp = cur.get("test_fp_per_1k")

        if any(v is None for v in [r_m50, r_m95, r_fp, c_m50, c_m95, c_fp]):
            print(f"Skipping Table IV for {det_name}: incomplete data.")
            continue

        delta_fp = c_fp - r_fp
        delta_str = f"{delta_fp:+.2f}"

        old_target = f"\\multirow{{2}}{{*}}{{{det_name}}}\n        & Random     & -- & -- & -- & \\multirow{{2}}{{*}}{{--}} \\\\\n        & Hard-Mined & -- & -- & -- & \\\\"
        new_target = f"\\multirow{{2}}{{*}}{{{det_name}}}\n        & Random     & {r_m50:.4f} & {r_m95:.4f} & {r_fp:.2f} & \\multirow{{2}}{{*}}{{{delta_str}}} \\\\\n        & Hard-Mined & {c_m50:.4f} & {c_m95:.4f} & {c_fp:.2f} & \\\\"

        if old_target in tex:
            tex = tex.replace(old_target, new_target, 1)
            print(f"  [TABLE IV] Successfully updated {det_name}")
        else:
            # Maybe already populated, try regex or pattern
            print(f"  [TABLE IV] Old target not found (may already be updated) for {det_name}")

    tex = tex.replace("sample counts strictly matched per architecture ($N{=}4{,}001$). RQ2 training in progress.",
                      "sample counts strictly matched per architecture ($N{=}4{,}001$).")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] Table IV written to {tex_path}")

    if PDFLATEX_PATH.exists():
        ms_dir = REPO_ROOT / "docs" / "manuscript"
        print("[PDF] Recompiling docs/manuscript/main.pdf (Pass 1)...")
        subprocess.run([str(PDFLATEX_PATH), "-interaction=nonstopmode", "main.tex"], cwd=str(ms_dir))
        print("[PDF] Recompiling docs/manuscript/main.pdf (Pass 2)...")
        subprocess.run([str(PDFLATEX_PATH), "-interaction=nonstopmode", "main.tex"], cwd=str(ms_dir))
        print("[OK] PDF recompilation complete.")

def main():
    args = parse_args()
    summary_file = REPO_ROOT / "runs" / ("rq2_curation_summary.json" if args.seed == 42 else f"rq2_curation_summary_seed{args.seed}.json")
    rq2_summary = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                rq2_summary = json.load(f)
        except Exception:
            rq2_summary = {}

    target_keys = [k.strip() for k in args.models.split(",")]
    target_models = [m for m in MODELS if m["key"] in target_keys]

    if args.populate_only:
        update_manuscript_table_iv(rq2_summary)
        return

    for m in target_models:
        key = m["key"]
        base_w = m["base_weights"]

        print(f"\n============================================================")
        print(f"  PROCESSING RQ2 FOR: {key.upper()} (SEED {args.seed})")
        print(f"============================================================")

        rand_res = load_rq1_random_result(key, seed=args.seed)
        if rand_res is None:
            print(f"[WARN] Could not find RQ1 40% run for {key}. Please verify {key}_sweep_summary.json.")
            continue
        print(f"  Random Baseline (40%): mAP@50={rand_res['test_map50']:.4f}, mAP@50:95={rand_res['test_map50_95']:.4f}, FP/1k={rand_res['test_fp_per_1k']:.2f}")

        ckpt_path, train_time = train_curated_model(key, base_w, args)

        cur_res = evaluate_curated_model(key, ckpt_path, train_time, args)

        delta_fp = cur_res["test_fp_per_1k"] - rand_res["test_fp_per_1k"]
        rq2_summary[key] = {
            "model": key,
            "target_count": 1600,
            "total_frames": 4001,
            "random": rand_res,
            "curated": cur_res,
            "delta_fp_per_1k": round(delta_fp, 2),
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(rq2_summary, f, indent=2)
        print(f"[OK] Incremental RQ2 summary saved to {summary_file}")

    if args.seed == 42:
        update_manuscript_table_iv(rq2_summary)
    print("\n[ALL RQ2 TASKS COMPLETED SUCCESSFULLY!]")

if __name__ == "__main__":
    main()
