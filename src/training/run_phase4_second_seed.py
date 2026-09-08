"""
run_phase4_second_seed.py - Multi-Seed (Seed 43) RQ1 Sweep Runner & Aggregator.

Protocol:
- Models: YOLO11n, YOLO26n, YOLO12n, YOLOv10n
- All 5 ratio splits (0%, 20%, 40%, 60%, 80%) per model (20 runs total)
- Seed: 43 (frozen protocol: 100 epochs, batch 16, imgsz 640, AdamW, close_mosaic 10)
- Output dirs: runs/{model_stem}_ratio_sweep_seed43
- Aggregates Seed 42 and Seed 43 metrics into mean +- std
- Updates Table III in docs/manuscript/main.tex and recompiles main.pdf
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PDFLATEX_PATH = Path(r"C:\Users\omarb\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")

MODELS = [
    "yolo11n.pt",
    "yolo26n.pt",
    "yolo12n.pt",
    "yolov10n.pt",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 Multi-Seed Sweep Runner")
    parser.add_argument("--models", type=str, default="yolo11n,yolo26n,yolo12n,yolov10n", help="Comma-separated model stems to run")
    parser.add_argument("--seed", type=int, default=43, help="Second seed (default: 43)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing seed 42 & 43 results")
    return parser.parse_args()

def run_model_sweep(model_file, seed, device):
    stem = Path(model_file).stem
    project_dir = REPO_ROOT / "runs" / f"{stem}_ratio_sweep_seed{seed}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "training" / "train_yolo_sweep.py"),
        "--model", str(REPO_ROOT / model_file),
        "--seed", str(seed),
        "--project", str(project_dir),
        "--splits", "all",
        "--device", str(device)
    ]
    print(f"\n{'='*70}")
    print(f"  STARTING SEED {seed} SWEEP FOR: {stem.upper()}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if ret.returncode != 0:
        raise RuntimeError(f"Sweep failed for {stem} with return code {ret.returncode}")

def load_sweep_json(project_dir, stem):
    summary_file = project_dir / f"{stem}_sweep_summary.json"
    if not summary_file.exists():
        return None
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)

def aggregate_seeds(seed1_data, seed2_data):
    """Compute mean and std for test metrics across two seeds."""
    r1_map = {r["split"]: r for r in seed1_data.get("runs", [])}
    r2_map = {r["split"]: r for r in seed2_data.get("runs", [])}
    
    splits = ["train_00_pos_only", "train_20_low_neg", "train_40_mod_neg", "train_60_high_neg", "train_80_max_neg"]
    ratio_labels = {"train_00_pos_only": "0%", "train_20_low_neg": "20%", "train_40_mod_neg": "40%", "train_60_high_neg": "60%", "train_80_max_neg": "80%"}
    
    aggregated = []
    for sp in splits:
        if sp not in r1_map or sp not in r2_map:
            continue
        v1 = r1_map[sp]
        v2 = r2_map[sp]
        
        m50_mean = (v1["test_map50"] + v2["test_map50"]) / 2.0
        m50_std = abs(v1["test_map50"] - v2["test_map50"]) / (2**0.5)
        
        m95_mean = (v1["test_map50_95"] + v2["test_map50_95"]) / 2.0
        m95_std = abs(v1["test_map50_95"] - v2["test_map50_95"]) / (2**0.5)
        
        p_mean = (v1["test_precision"] + v2["test_precision"]) / 2.0
        r_mean = (v1["test_recall"] + v2["test_recall"]) / 2.0
        
        fp_mean = (v1["test_fp_per_1k"] + v2["test_fp_per_1k"]) / 2.0
        fp_std = abs(v1["test_fp_per_1k"] - v2["test_fp_per_1k"]) / (2**0.5)
        
        raw_fps_avg = (v1["total_test_fps"] + v2["total_test_fps"]) / 2.0
        
        aggregated.append({
            "split": sp,
            "ratio": ratio_labels[sp],
            "map50_mean": round(m50_mean, 4),
            "map50_std": round(m50_std, 4),
            "map50_95_mean": round(m95_mean, 4),
            "map50_95_std": round(m95_std, 4),
            "precision_mean": round(p_mean, 4),
            "recall_mean": round(r_mean, 4),
            "fp_per_1k_mean": round(fp_mean, 2),
            "fp_per_1k_std": round(fp_std, 2),
            "raw_fps_avg": round(raw_fps_avg, 1),
            "seed42": v1,
            "seed43": v2,
        })
    return aggregated

def main():
    args = parse_args()
    target_stems = [s.strip() for s in args.models.split(",")]
    
    if not args.aggregate_only:
        for m_file in MODELS:
            stem = Path(m_file).stem
            if stem not in target_stems:
                continue
            run_model_sweep(m_file, seed=args.seed, device=args.device)
            
    # Aggregation
    print("\n" + "="*70)
    print("  COMPUTING MULTI-SEED (SEED 42 vs 43) AGGREGATIONS")
    print("="*70 + "\n")
    
    multi_seed_summary = {}
    for m_file in MODELS:
        stem = Path(m_file).stem
        s42_dir = REPO_ROOT / "runs" / f"{stem}_ratio_sweep"
        s43_dir = REPO_ROOT / "runs" / f"{stem}_ratio_sweep_seed43"
        
        d42 = load_sweep_json(s42_dir, stem)
        d43 = load_sweep_json(s43_dir, stem)
        
        if not d42 or not d43:
            print(f"[WARN] Incomplete seeds for {stem}. Seed 42: {d42 is not None}, Seed 43: {d43 is not None}")
            continue
            
        agg = aggregate_seeds(d42, d43)
        multi_seed_summary[stem] = agg
        print(f"Aggregated {len(agg)} splits for {stem}.")
        
    out_file = REPO_ROOT / "runs" / "multi_seed_rq1_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(multi_seed_summary, f, indent=2)
    print(f"\n[OK] Multi-seed summary written to: {out_file}")

if __name__ == "__main__":
    main()
