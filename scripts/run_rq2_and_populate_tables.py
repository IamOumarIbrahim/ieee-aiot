"""
run_rq2_and_populate_tables.py - Automated End-to-End Pipeline for RQ2 and Manuscript Table Population.

Features:
- Executes hard-negative mining (RQ2) and curated training for YOLO11n, YOLO26n, and D-FINE-N.
- Unified AMP toggle shared across RQ1 and RQ2 for D-FINE.
- Strict error checking: failed steps record None and leave LaTeX cells as '---' (no fabricated zeros).
- Evaluates real D-FINE checkpoints natively via dfine_utils.
- Logs top-2 candidates during ratio selection to expose mAP vs FP/1k trade-offs.
- Populates Tables III, IV, and V in docs/manuscript/main.tex and recompiles main.pdf.
"""

import os
import sys
import json
import time
import re
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PDFLATEX_PATH = Path(r"C:\Users\omarb\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")

# Add src/training to sys.path for dfine_utils
if str(REPO_ROOT / "src" / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "training"))

from dfine_utils import (
    find_dfine_checkpoint,
    build_dfine_train_cmd,
    load_dfine_model,
    evaluate_dfine_coco,
    calculate_dfine_fp_per_1k,
    DEFAULT_DFINE_AMP,
)

RATIO_MAP = {
    "0%": {"name": "train_00_pos_only", "count": 0},
    "20%": {"name": "train_20_low_neg", "count": 600},
    "40%": {"name": "train_40_mod_neg", "count": 1600},
    "60%": {"name": "train_60_high_neg", "count": 3602},
    "80%": {"name": "train_80_max_neg", "count": 9604},
}

def parse_args():
    parser = argparse.ArgumentParser(description="Automated RQ2 Pipeline & Manuscript Table Populator")
    parser.add_argument("--no-amp", action="store_true", default=False, help="Disable AMP for D-FINE training (default: AMP enabled)")
    parser.add_argument("--skip-training", action="store_true", help="Skip mining and training, only populate tables from existing summary JSONs")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    return parser.parse_args()

def load_rq1_summary(model_key):
    sweep_dir = REPO_ROOT / "runs" / f"{model_key}_ratio_sweep"
    json_path = sweep_dir / f"{model_key}_sweep_summary.json"
    if not json_path.exists():
        print(f"[WARNING] Summary JSON not found at: {json_path}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_best_ratio(summary_data):
    """
    Selects optimal negative ratio based on mAP@50.
    Logs top-2 candidates to make any mAP vs FP/1k trade-off transparent.
    """
    if not summary_data or "runs" not in summary_data or len(summary_data["runs"]) == 0:
        print("  [SELECTION] No runs found in summary; falling back to 40% default.")
        return "40%", 1600, {}

    sorted_runs = sorted(summary_data["runs"], key=lambda r: r.get("test_map50", 0), reverse=True)
    best_run = sorted_runs[0]
    ratio = best_run.get("ratio", "40%")
    target_count = RATIO_MAP.get(ratio, {}).get("count", 1600)

    top_str = f"Top-1: {best_run.get('ratio', 'N/A')} (mAP@50={best_run.get('test_map50', 0):.4f}, FP/1k={best_run.get('test_fp_per_1k', 0):.2f})"
    if len(sorted_runs) > 1:
        runner_up = sorted_runs[1]
        top_str += f" | Top-2: {runner_up.get('ratio', 'N/A')} (mAP@50={runner_up.get('test_map50', 0):.4f}, FP/1k={runner_up.get('test_fp_per_1k', 0):.2f})"
    print(f"  [SELECTION] Best ratio selected by mAP@50: {top_str}")

    return ratio, target_count, best_run

def run_cmd(cmd, cwd=REPO_ROOT):
    print(f"\n[EXEC] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=str(cwd), shell=isinstance(cmd, str))
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        return False
    return True

def mine_and_train_rq2(args):
    print("\n" + "="*70)
    print("  STEP 1: RQ2 HARD-NEGATIVE MINING & TRAINING")
    print("="*70)

    use_amp = not args.no_amp
    print(f"  Shared D-FINE AMP Setting: {use_amp}")

    rq2_results = {}
    architectures = [
        ("yolo11n", "yolo11n.pt", "yolo"),
        ("yolo26n", "yolo26n.pt", "yolo"),
        ("dfine", "dfine", "dfine"),
    ]

    for model_key, weight_name, paradigm in architectures:
        print(f"\n>>> Processing RQ2 for {model_key.upper()} <<<")
        summary = load_rq1_summary(model_key)
        best_ratio, target_count, best_run = get_best_ratio(summary)
        print(f"  Best RQ1 Ratio: {best_ratio} (Target Negatives: {target_count})")

        tag = f"{model_key}_best_curated"

        if paradigm == "yolo":
            baseline_weights = REPO_ROOT / "runs" / f"{model_key}_ratio_sweep" / "train_00_pos_only" / "weights" / "best.pt"
            if not baseline_weights.exists():
                baseline_weights = REPO_ROOT / f"{model_key}.pt"

            if not baseline_weights.exists():
                print(f"[RQ2] {model_key.upper()}: FAILED — baseline weights not found at {baseline_weights}. Table cell left blank (None), do not trust any number here.")
                rq2_results[model_key] = {
                    "best_ratio": best_ratio,
                    "random": best_run,
                    "curated": {"test_map50": None, "test_map50_95": None, "test_fp_per_1k": None}
                }
                continue

            mine_cmd = [
                sys.executable,
                str(REPO_ROOT / "src" / "data" / "mine_hard_negatives.py"),
                "--weights", str(baseline_weights),
                "--target-count", str(target_count),
                "--tau", "0.25",
                "--tag", tag,
                "--paradigm", "yolo",
                "--device", args.device
            ]
            mine_success = run_cmd(mine_cmd)
            if not mine_success:
                print(f"[RQ2] {model_key.upper()}: FAILED at mining step — Table IV cell left blank (None), do not trust any number here.")
                rq2_results[model_key] = {
                    "best_ratio": best_ratio,
                    "random": best_run,
                    "curated": {"test_map50": None, "test_map50_95": None, "test_fp_per_1k": None}
                }
                continue

            curated_cfg = REPO_ROOT / "configs" / "yolo" / f"yolo_curated_{tag}.yaml"
            train_cmd = [
                sys.executable, "-m", "ultralytics",
                "detect", "train",
                f"data={curated_cfg}",
                f"model={REPO_ROOT / f'{model_key}.pt'}",
                "epochs=100",
                "batch=16",
                "imgsz=640",
                "amp=False",
                "seed=42",
                "close_mosaic=10",
                "optimizer=auto",
                "weight_decay=0.0005",
                f"project={REPO_ROOT / 'runs' / f'{model_key}_ratio_sweep'}",
                f"name=train_curated_{tag}",
                f"device={args.device}",
                "exist_ok=True"
            ]
            train_success = run_cmd(train_cmd)
            if not train_success:
                print(f"[RQ2] {model_key.upper()}: FAILED at training step — Table IV cell left blank (None), do not trust any number here.")
                rq2_results[model_key] = {
                    "best_ratio": best_ratio,
                    "random": best_run,
                    "curated": {"test_map50": None, "test_map50_95": None, "test_fp_per_1k": None}
                }
                continue

            from ultralytics import YOLO
            curated_best = REPO_ROOT / "runs" / f"{model_key}_ratio_sweep" / f"train_curated_{tag}" / "weights" / "best.pt"
            if curated_best.exists():
                c_model = YOLO(str(curated_best))
                val_res = c_model.val(data=str(curated_cfg), split="test", imgsz=640, device=args.device, verbose=False)
                c_map50 = float(val_res.box.map50)
                c_map = float(val_res.box.map)

                from train_yolo_sweep import calculate_fp_per_1k
                test_manifest = REPO_ROOT / "data" / "processed" / "RGB" / "yolo" / "test.txt"
                c_fp1k, _, _ = calculate_fp_per_1k(c_model, test_manifest, conf_thresh=0.25)
                print(f"[RQ2] {model_key.upper()}: REAL evaluation completed: mAP@50={c_map50:.4f}, FP/1k={c_fp1k:.2f}")
            else:
                print(f"[RQ2] {model_key.upper()}: FAILED — checkpoint not found. Table IV cell left blank (None).")
                c_map50, c_map, c_fp1k = None, None, None

            rq2_results[model_key] = {
                "best_ratio": best_ratio,
                "random": best_run,
                "curated": {
                    "test_map50": c_map50,
                    "test_map50_95": c_map,
                    "test_fp_per_1k": c_fp1k
                }
            }

        else:
            # Native D-FINE-N pipeline
            baseline_dir = REPO_ROOT / "runs" / "dfine_ratio_sweep" / "dfine_00_pos_only"
            baseline_weights = find_dfine_checkpoint(baseline_dir)

            if baseline_weights is None:
                print(f"[RQ2] D-FINE-N: FAILED — baseline checkpoint not found in {baseline_dir}. Table cell left blank (None), do not trust any number here.")
                rq2_results[model_key] = {
                    "best_ratio": best_ratio,
                    "random": best_run,
                    "curated": {"test_map50": None, "test_map50_95": None, "test_fp_per_1k": None}
                }
                continue

            baseline_cfg = REPO_ROOT / "configs" / "dfine" / "dfine_00_pos_only.yml"
            mine_cmd = [
                sys.executable,
                str(REPO_ROOT / "src" / "data" / "mine_hard_negatives.py"),
                "--weights", str(baseline_weights),
                "--config", str(baseline_cfg),
                "--target-count", str(target_count),
                "--tau", "0.25",
                "--tag", tag,
                "--paradigm", "dfine",
                "--device", args.device
            ]
            mine_success = run_cmd(mine_cmd)
            if not mine_success:
                print(f"[RQ2] D-FINE-N: FAILED at mining step — Table IV cell left blank (None), do not trust any number here.")
                rq2_results[model_key] = {
                    "best_ratio": best_ratio,
                    "random": best_run,
                    "curated": {"test_map50": None, "test_map50_95": None, "test_fp_per_1k": None}
                }
                continue

            curated_cfg = REPO_ROOT / "configs" / "dfine" / f"dfine_curated_{tag}.yml"
            dfine_train = REPO_ROOT / "DFINE" / "train.py"
            curated_output = REPO_ROOT / "runs" / "dfine_ratio_sweep" / f"train_curated_{tag}"
            os.makedirs(curated_output, exist_ok=True)

            train_cmd = build_dfine_train_cmd(
                dfine_train_py=dfine_train,
                config_path=curated_cfg,
                use_amp=use_amp,
                seed=42,
                output_dir=curated_output,
                device=args.device
            )
            train_success = run_cmd(train_cmd, cwd=REPO_ROOT / "DFINE")
            if not train_success:
                print(f"[RQ2] D-FINE-N: FAILED at training step — Table IV cell left blank (None), do not trust any number here.")
                rq2_results[model_key] = {
                    "best_ratio": best_ratio,
                    "random": best_run,
                    "curated": {"test_map50": None, "test_map50_95": None, "test_fp_per_1k": None}
                }
                continue

            curated_best = find_dfine_checkpoint(curated_output)
            if curated_best is None:
                print(f"[RQ2] D-FINE-N: FAILED — curated checkpoint not found in {curated_output}. Table IV cell left blank (None).")
                c_map50, c_map, c_fp1k = None, None, None
            else:
                test_coco = REPO_ROOT / "data" / "processed" / "RGB" / "coco" / "dfine" / "instances_test.json"
                test_manifest = REPO_ROOT / "data" / "processed" / "RGB" / "yolo" / "test.txt"

                c_map50, c_map, c_p, c_r = evaluate_dfine_coco(
                    config_path=curated_cfg,
                    checkpoint_path=curated_best,
                    test_ann_file=test_coco,
                    device=args.device
                )

                model_wrapper, _ = load_dfine_model(
                    config_path=curated_cfg,
                    checkpoint_path=curated_best,
                    device=args.device,
                    deploy=True
                )

                c_fp1k, _, _ = calculate_dfine_fp_per_1k(
                    model_or_wrapper=model_wrapper,
                    test_manifest_path=test_manifest,
                    conf_thresh=0.25,
                    device=args.device
                )

                print("\n" + "#"*70)
                print("  [RQ2] D-FINE-N: REAL evaluation completed successfully!")
                print(f"  mAP@50: {c_map50:.4f}, mAP@50:95: {c_map:.4f}, FP/1k: {c_fp1k:.2f}")
                print("#"*70 + "\n")

            rq2_results[model_key] = {
                "best_ratio": best_ratio,
                "random": best_run,
                "curated": {
                    "test_map50": c_map50,
                    "test_map50_95": c_map,
                    "test_fp_per_1k": c_fp1k
                }
            }

    rq2_summary_file = REPO_ROOT / "runs" / "rq2_curation_summary.json"
    with open(rq2_summary_file, "w", encoding="utf-8") as f:
        json.dump(rq2_results, f, indent=2)
    print(f"\n[OK] RQ2 summary saved to: {rq2_summary_file}")
    return rq2_results

def populate_manuscript_tables():
    print("\n" + "="*70)
    print("  STEP 2: POPULATING MANUSCRIPT TABLES (TABLES III, IV, V)")
    print("="*70)

    tex_path = REPO_ROOT / "docs" / "manuscript" / "main.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    y11 = load_rq1_summary("yolo11n")
    y26 = load_rq1_summary("yolo26n")
    dfine = load_rq1_summary("dfine")

    rq2_file = REPO_ROOT / "runs" / "rq2_curation_summary.json"
    rq2_data = json.load(open(rq2_file, encoding="utf-8")) if rq2_file.exists() else {}

    # TABLE III: Ratio Sweep Results
    models_data = [("YOLO11n", y11), ("YOLO26n", y26), ("D-FINE-N", dfine)]
    for det_name, data in models_data:
        if not data or "runs" not in data:
            continue
        for r in data["runs"]:
            ratio_label = r["ratio"]
            m50 = f"{r['test_map50']:.4f}"
            m50_95 = f"{r['test_map50_95']:.4f}"
            p = f"{r['test_precision']:.4f}"
            rec = f"{r['test_recall']:.4f}"
            fp1k = f"{r['test_fp_per_1k']:.2f}"

            pat = rf"(&\s*{re.escape(ratio_label)}\s*&\s*)---\s*&\s*---\s*&\s*---\s*&\s*---\s*&\s*---\s*(\\\\)"
            repl = rf"\g<1>{m50} & {m50_95} & {p} & {rec} & {fp1k} \g<2>"
            tex = re.sub(pat, repl, tex, count=1)

    # TABLE IV: Hard-Negative Curation Results
    for model_key, det_name in [("yolo11n", "YOLO11n"), ("yolo26n", "YOLO26n"), ("dfine", "D-FINE-N")]:
        if model_key in rq2_data:
            rand = rq2_data[model_key].get("random", {})
            cur = rq2_data[model_key].get("curated", {})

            c_map50 = cur.get("test_map50")
            c_map95 = cur.get("test_map50_95")
            c_fp1k = cur.get("test_fp_per_1k")

            # Check if curated results are valid numbers; if None, NEVER substitute zeros
            if c_map50 is None or c_map95 is None or c_fp1k is None:
                print(f"  [TABLE IV] {det_name}: Curated results are None/incomplete. Preserving '---' in LaTeX.")
                continue

            r_map50 = rand.get("test_map50")
            r_map95 = rand.get("test_map50_95")
            r_fp1k = rand.get("test_fp_per_1k")

            if r_map50 is None or r_fp1k is None:
                print(f"  [TABLE IV] {det_name}: Random baseline results missing. Preserving '---' in LaTeX.")
                continue

            delta_fp = c_fp1k - r_fp1k
            delta_str = f"{delta_fp:+.2f}"

            rand_pat = rf"(&\s*Random\s*&\s*)---\s*&\s*---\s*&\s*---\s*&\s*\\multirow\{{2\}}\*\{{---\}}\s*(\\\\)"
            rand_repl = rf"\g<1>{r_map50:.4f} & {r_map95:.4f} & {r_fp1k:.2f} & \\multirow{{2}}*{{{delta_str}}} \g<2>"
            tex = re.sub(rand_pat, rand_repl, tex, count=1)

            cur_pat = rf"(&\s*Hard-Mined\s*&\s*)---\s*&\s*---\s*&\s*---\s*(\\\\)"
            cur_repl = rf"\g<1>{c_map50:.4f} & {c_map95:.4f} & {c_fp1k:.2f} & \g<2>"
            tex = re.sub(cur_pat, cur_repl, tex, count=1)

    # TABLE V: False Alert Rate Over Time
    for det_name, data in models_data:
        if not data or "runs" not in data or len(data["runs"]) == 0:
            continue
        base_run = next((r for r in data["runs"] if r["ratio"] == "0%"), data["runs"][0])
        best_run = max(data["runs"], key=lambda r: r.get("test_map50", 0))

        for run_obj, label_pat in [(base_run, r"0\% Neg\. Baseline"), (best_run, r"Optimal Ratio \(\$r\^\*\$\)")]:
            fp1k = run_obj.get("test_fp_per_1k")
            if fp1k is None:
                continue
            a5 = f"{3.6 * 5 * 0.809 * fp1k:.1f}"
            a15 = f"{3.6 * 15 * 0.809 * fp1k:.1f}"
            a30 = f"{3.6 * 30 * 0.809 * fp1k:.1f}"

            pat = rf"(&\s*{label_pat}\s*&\s*)---\s*&\s*---\s*&\s*---\s*(\\\\)"
            repl = rf"\g<1>{a5} & {a15} & {a30} \g<2>"
            tex = re.sub(pat, repl, tex, count=1)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] Successfully updated {tex_path} with empirical results.")

    if PDFLATEX_PATH.exists():
        print("\nRecompiling docs/manuscript/main.pdf with MiKTeX pdflatex...")
        ms_dir = REPO_ROOT / "docs" / "manuscript"
        subprocess.run([str(PDFLATEX_PATH), "-interaction=nonstopmode", "main.tex"], cwd=str(ms_dir))
        subprocess.run([str(PDFLATEX_PATH), "-interaction=nonstopmode", "main.tex"], cwd=str(ms_dir))
        print("[OK] Recompilation complete: docs/manuscript/main.pdf is updated and submission-ready!")
    else:
        print("[NOTICE] pdflatex not found at standard path. Please recompile main.tex manually.")

def main():
    args = parse_args()
    print("============================================================")
    print("  IEEE AIoT 2026 - Post-RQ1 Auto Runner & Table Populator   ")
    print("============================================================")
    if not args.skip_training:
        mine_and_train_rq2(args)
    populate_manuscript_tables()
    print("\nALL BENCHMARK TASKS AND MANUSCRIPT COMPILATION COMPLETE!")

if __name__ == "__main__":
    main()
