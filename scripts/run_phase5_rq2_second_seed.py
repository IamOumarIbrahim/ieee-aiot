"""
run_phase5_rq2_second_seed.py - RQ2 (Option 3) Multi-Seed Runner & Table IV Aggregator.

Executes after Option 2 finishes:
1. Trains Curated-40% models with seed=43 for YOLO11n, YOLO26n, YOLO12n, YOLOv10n.
2. Evaluates held-out test splits.
3. Loads seed=42 and seed=43 RQ2 summaries.
4. Computes cross-seed mean +- std for Random and Hard-Mined splits.
5. Updates Table IV in docs/manuscript/main.tex and recompiles main.pdf (<= 6 pages).
6. Pushes to GitHub origin main.
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PDFLATEX_PATH = Path(r"C:\Users\omarb\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")

MODELS = ["yolo11n", "yolo26n", "yolo12n", "yolov10n"]

def parse_args():
    parser = argparse.ArgumentParser(description="RQ2 Second-Seed Runner")
    parser.add_argument("--models", type=str, default="yolo11n,yolo26n,yolo12n,yolov10n", help="Comma-separated model keys")
    parser.add_argument("--seed", type=int, default=43, help="Random seed (default: 43)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate and update paper")
    return parser.parse_args()

def run_curated_seed(models_str, seed, device):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_rq2_curated_sweep.py"),
        "--models", models_str,
        "--seed", str(seed),
        "--device", str(device)
    ]
    print(f"\n{'='*70}")
    print(f"  LAUNCHING RQ2 CURATED SEED {seed} TRAINING")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if ret.returncode != 0:
        raise RuntimeError(f"RQ2 seed {seed} training failed with exit code {ret.returncode}")

def aggregate_rq2_seeds(s42_data, s43_data):
    """Aggregate Seed 42 and Seed 43 results for Table IV."""
    agg = {}
    for m in MODELS:
        if m not in s42_data or m not in s43_data:
            continue
        v1 = s42_data[m]
        v2 = s43_data[m]

        r1, r2 = v1["random"], v2["random"]
        c1, c2 = v1["curated"], v2["curated"]

        # Random means & stds
        r_m50_mean = (r1["test_map50"] + r2["test_map50"]) / 2.0
        r_m50_std = abs(r1["test_map50"] - r2["test_map50"]) / (2**0.5)

        r_m95_mean = (r1["test_map50_95"] + r2["test_map50_95"]) / 2.0
        r_m95_std = abs(r1["test_map50_95"] - r2["test_map50_95"]) / (2**0.5)

        r_fp_mean = (r1["test_fp_per_1k"] + r2["test_fp_per_1k"]) / 2.0
        r_fp_std = abs(r1["test_fp_per_1k"] - r2["test_fp_per_1k"]) / (2**0.5)

        # Curated means & stds
        c_m50_mean = (c1["test_map50"] + c2["test_map50"]) / 2.0
        c_m50_std = abs(c1["test_map50"] - c2["test_map50"]) / (2**0.5)

        c_m95_mean = (c1["test_map50_95"] + c2["test_map50_95"]) / 2.0
        c_m95_std = abs(c1["test_map50_95"] - c2["test_map50_95"]) / (2**0.5)

        c_fp_mean = (c1["test_fp_per_1k"] + c2["test_fp_per_1k"]) / 2.0
        c_fp_std = abs(c1["test_fp_per_1k"] - c2["test_fp_per_1k"]) / (2**0.5)

        delta_fp_mean = c_fp_mean - r_fp_mean
        pct_delta = ((c_fp_mean - r_fp_mean) / r_fp_mean) * 100.0

        agg[m] = {
            "model": m,
            "mined": v1.get("mined_count", 0),
            "fill": v1.get("backfill_count", 0),
            "random": {
                "map50_mean": round(r_m50_mean, 4),
                "map50_std": round(r_m50_std, 4),
                "map50_95_mean": round(r_m95_mean, 4),
                "map50_95_std": round(r_m95_std, 4),
                "fp_per_1k_mean": round(r_fp_mean, 2),
                "fp_per_1k_std": round(r_fp_std, 2),
            },
            "curated": {
                "map50_mean": round(c_m50_mean, 4),
                "map50_std": round(c_m50_std, 4),
                "map50_95_mean": round(c_m95_mean, 4),
                "map50_95_std": round(c_m95_std, 4),
                "fp_per_1k_mean": round(c_fp_mean, 2),
                "fp_per_1k_std": round(c_fp_std, 2),
            },
            "pct_delta": round(pct_delta, 1),
            "delta_fp_per_1k": round(delta_fp_mean, 2),
        }
    return agg

def main():
    args = parse_args()
    if not args.aggregate_only:
        run_curated_seed(args.models, args.seed, args.device)

    s42_file = REPO_ROOT / "runs" / "rq2_curation_summary.json"
    s43_file = REPO_ROOT / "runs" / f"rq2_curation_summary_seed{args.seed}.json"

    if not s42_file.exists() or not s43_file.exists():
        print(f"[ERROR] Required summaries missing: s42={s42_file.exists()}, s43={s43_file.exists()}")
        return

    with open(s42_file) as f:
        s42 = json.load(f)
    with open(s43_file) as f:
        s43 = json.load(f)

    agg = aggregate_rq2_seeds(s42, s43)
    out_file = REPO_ROOT / "runs" / "multi_seed_rq2_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[OK] Aggregated RQ2 summary written to: {out_file}")

if __name__ == "__main__":
    main()
