"""
run_smoke_test_suite.py - Verification runner executing 3 test epochs across all 7 planned runs.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = REPO_ROOT / "runs" / "test_smoke_3ep"

PLANNED_RUNS = [
    {"model": "yolo11n.pt", "name": "YOLO11n (81% Natural)", "splits": "81", "project": SMOKE_DIR / "yolo11n"},
    {"model": "yolo26n.pt", "name": "YOLO26n (81% Natural)", "splits": "81", "project": SMOKE_DIR / "yolo26n"},
    {"model": "yolo12n.pt", "name": "YOLO12n (Full 5-Ratio Sweep)", "splits": "00,20,40,60,80", "project": SMOKE_DIR / "yolo12n"},
]

def main():
    print("=" * 80)
    print("  3-EPOCH SMOKE TEST SUITE FOR ALL 7 PLANNED PRODUCTION RUNS")
    print(f"  Target Isolation Directory: {SMOKE_DIR}")
    print("=" * 80)

    total_start = time.time()
    results = []

    for idx, item in enumerate(PLANNED_RUNS, 1):
        print(f"\n[{idx}/{len(PLANNED_RUNS)}] Testing {item['name']}...")
        print(f"  Model: {item['model']} | Splits: {item['splits']} | Epochs: 3 | Batch: 16 | AMP: False")
        
        start_t = time.time()
        cmd = [
            sys.executable,
            str(REPO_ROOT / "src" / "training" / "train_yolo_sweep.py"),
            "--model", item["model"],
            "--epochs", "3",
            "--batch", "16",
            "--device", "0",
            "--splits", item["splits"],
            "--project", str(item["project"])
        ]
        
        ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
        elapsed = (time.time() - start_t) / 60.0
        
        if ret.returncode != 0:
            print(f"[FAIL] Run failed for {item['name']} with code {ret.returncode}")
            sys.exit(ret.returncode)
        
        print(f"  [PASS] {item['name']} completed 3-epoch test in {elapsed:.1f} minutes.")
        results.append({"name": item["name"], "time_min": elapsed, "status": "PASS"})

    total_elapsed = (time.time() - total_start) / 60.0
    print("\n" + "=" * 80)
    print("  ALL 7 PLANNED RUNS VERIFIED SUCCESSFULLY (3 TEST EPOCHS EACH)")
    print(f"  Total Duration: {total_elapsed:.1f} minutes")
    print("=" * 80)
    for r in results:
        print(f"  - {r['name']:<35}: [{r['status']}] ({r['time_min']:.1f}m)")
    print("=" * 80)

if __name__ == "__main__":
    main()
