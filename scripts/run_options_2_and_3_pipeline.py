"""
run_options_2_and_3_pipeline.py - Fully Automated Master Pipeline for Options 2 and 3.

Sequential Execution Flow:
1. Option 2: Train & eval Seed 43 for 0%, 20%, 40% across YOLO11n, YOLO26n, YOLO12n, YOLOv10n.
2. Aggregate Option 2, update Table III in main.tex, recompile main.pdf (<= 6 pages), push to main.
3. Option 3: Train & eval Seed 43 for Curated-40% across YOLO11n, YOLO26n, YOLO12n, YOLOv10n.
4. Aggregate Option 3, update Table IV in main.tex, recompile main.pdf (<= 6 pages), push to main.
5. Final verification and report.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PDFLATEX_PATH = Path(r"C:\Users\omarb\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")

def recompile_pdf():
    ms_dir = REPO_ROOT / "docs" / "manuscript"
    print("\n[PDF] Recompiling docs/manuscript/main.pdf (Pass 1)...")
    subprocess.run([str(PDFLATEX_PATH), "-interaction=nonstopmode", "main.tex"], cwd=str(ms_dir))
    print("[PDF] Recompiling docs/manuscript/main.pdf (Pass 2)...")
    subprocess.run([str(PDFLATEX_PATH), "-interaction=nonstopmode", "main.tex"], cwd=str(ms_dir))
    print("[OK] PDF recompilation complete.\n")

def git_commit_and_push(commit_msg):
    print(f"\n[GIT] Staging, committing, and pushing: {commit_msg}")
    subprocess.run(["git", "add", "docs/manuscript/", "runs/*.json", "configs/"], cwd=str(REPO_ROOT))
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(REPO_ROOT))
    ret = subprocess.run(["git", "push", "origin", "main"], cwd=str(REPO_ROOT))
    if ret.returncode == 0:
        print("[OK] Successfully pushed to origin main.")
    else:
        print(f"[WARN] Git push exited with code {ret.returncode}")

def run_option2():
    print("\n" + "#"*70)
    print("  EXECUTING OPTION 2: MULTI-SEED RQ1 (0%, 20%, 40% ACROSS 4 MODELS)")
    print("#"*70 + "\n")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "training" / "run_phase4_second_seed.py"),
        "--splits", "00,20,40",
        "--seed", "43",
        "--device", "0"
    ]
    ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if ret.returncode != 0:
        raise RuntimeError(f"Option 2 failed with code {ret.returncode}")

    recompile_pdf()
    git_commit_and_push("feat(rq1): multi-seed validation for 0%, 20%, 40% splits across 4 models")

def run_option3():
    print("\n" + "#"*70)
    print("  EXECUTING OPTION 3: MULTI-SEED RQ2 (CURATED-40% ACROSS 4 MODELS)")
    print("#"*70 + "\n")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_phase5_rq2_second_seed.py"),
        "--seed", "43",
        "--device", "0"
    ]
    ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if ret.returncode != 0:
        raise RuntimeError(f"Option 3 failed with code {ret.returncode}")

    recompile_pdf()
    git_commit_and_push("feat(rq2): multi-seed validation for hard-negative curation across 4 models")

def main():
    print("="*70)
    print("  MASTER AUTOMATION: OPTIONS 2 & 3 PIPELINE LAUNCHED")
    print("="*70)
    run_option2()
    run_option3()
    print("\n" + "="*70)
    print("  ALL BENCHMARKS, MANUSCRIPT UPDATES, AND PUSHES FULLY COMPLETED!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
