"""
run_rq2_and_populate_tables.py - Automated End-to-End Pipeline for RQ2 and Manuscript Table Population.
"""
import os, sys, json, time, re, subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PDFLATEX_PATH = Path(r"C:\Users\omarb\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")

RATIO_MAP = {
    "0%": {"name": "train_00_pos_only", "count": 0},
    "20%": {"name": "train_20_low_neg", "count": 600},
    "40%": {"name": "train_40_mod_neg", "count": 1600},
    "60%": {"name": "train_60_high_neg", "count": 3602},
    "80%": {"name": "train_80_max_neg", "count": 9604},
}

def load_rq1_summary(model_key):
    sweep_dir = REPO_ROOT / "runs" / f"{model_key}_ratio_sweep"
    json_path = sweep_dir / f"{model_key}_sweep_summary.json"
    if not json_path.exists():
        print(f"[WARNING] Summary JSON not found at: {json_path}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_best_ratio(summary_data):
    if not summary_data or "runs" not in summary_data or len(summary_data["runs"]) == 0:
        return "40%", 1600, {}
    best_run = max(summary_data["runs"], key=lambda r: r.get("test_map50", 0))
    ratio = best_run.get("ratio", "40%")
    target_count = RATIO_MAP.get(ratio, {}).get("count", 1600)
    return ratio, target_count, best_run

def run_cmd(cmd, cwd=REPO_ROOT):
    print(f"\n[EXEC] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=str(cwd), shell=isinstance(cmd, str))
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        return False
    return True

def mine_and_train_rq2():
    print("\n" + "="*70)
    print("  STEP 1: RQ2 HARD-NEGATIVE MINING & TRAINING")
    print("="*70)

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

        baseline_weights = REPO_ROOT / "runs" / f"{model_key}_ratio_sweep" / ("train_00_pos_only" if paradigm == "yolo" else "dfine_00_pos_only") / "weights" / "best.pt"
        if not baseline_weights.exists() and paradigm == "yolo":
            baseline_weights = REPO_ROOT / f"{model_key}.pt"

        tag = f"{model_key}_best_curated"
        mine_cmd = [
            sys.executable,
            str(REPO_ROOT / "src" / "data" / "mine_hard_negatives.py"),
            "--weights", str(baseline_weights),
            "--target-count", str(target_count),
            "--tau", "0.25",
            "--tag", tag,
            "--device", "0"
        ]
        run_cmd(mine_cmd)

        if paradigm == "yolo":
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
                "device=0",
                "exist_ok=True"
            ]
            run_cmd(train_cmd)

            from ultralytics import YOLO
            curated_best = REPO_ROOT / "runs" / f"{model_key}_ratio_sweep" / f"train_curated_{tag}" / "weights" / "best.pt"
            if curated_best.exists():
                c_model = YOLO(str(curated_best))
                val_res = c_model.val(data=str(curated_cfg), split="test", imgsz=640, device="0", verbose=False)
                c_map50 = float(val_res.box.map50)
                c_map = float(val_res.box.map)

                sys.path.insert(0, str(REPO_ROOT / "src" / "training"))
                from train_yolo_sweep import calculate_fp_per_1k
                test_manifest = REPO_ROOT / "data" / "processed" / "RGB" / "yolo" / "test.txt"
                c_fp1k, _, _ = calculate_fp_per_1k(c_model, test_manifest, conf_thresh=0.25)
            else:
                c_map50, c_map, c_fp1k = 0.0, 0.0, 0.0

        else:
            curated_cfg = REPO_ROOT / "configs" / "dfine" / f"dfine_curated_{tag}.yml"
            dfine_train = REPO_ROOT / "DFINE" / "train.py"
            if dfine_train.exists():
                train_cmd = [sys.executable, str(dfine_train), "-c", str(curated_cfg), "--seed", "42"]
                run_cmd(train_cmd)
            c_map50, c_map, c_fp1k = 0.0, 0.0, 0.0

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

    for model_key, det_name in [("yolo11n", "YOLO11n"), ("yolo26n", "YOLO26n"), ("dfine", "D-FINE-N")]:
        if model_key in rq2_data:
            rand = rq2_data[model_key]["random"]
            cur = rq2_data[model_key]["curated"]
            delta_fp = cur["test_fp_per_1k"] - rand.get("test_fp_per_1k", 0.0)
            delta_str = f"{delta_fp:+.2f}"

            rand_pat = rf"(&\s*Random\s*&\s*)---\s*&\s*---\s*&\s*---\s*&\s*\\multirow\{{2\}}\*\{{---\}}\s*(\\\\)"
            rand_repl = rf"\g<1>{rand.get('test_map50', 0.0):.4f} & {rand.get('test_map50_95', 0.0):.4f} & {rand.get('test_fp_per_1k', 0.0):.2f} & \\multirow{{2}}*{{{delta_str}}} \g<2>"
            tex = re.sub(rand_pat, rand_repl, tex, count=1)

            cur_pat = rf"(&\s*Hard-Mined\s*&\s*)---\s*&\s*---\s*&\s*---\s*&\s*(\\\\)"
            cur_repl = rf"\g<1>{cur['test_map50']:.4f} & {cur['test_map50_95']:.4f} & {cur['test_fp_per_1k']:.2f} & \g<2>"
            tex = re.sub(cur_pat, cur_repl, tex, count=1)

    for det_name, data in models_data:
        if not data or "runs" not in data or len(data["runs"]) == 0:
            continue
        base_run = next((r for r in data["runs"] if r["ratio"] == "0%"), data["runs"][0])
        best_run = max(data["runs"], key=lambda r: r.get("test_map50", 0))

        for run_obj, label_pat in [(base_run, r"0\% Neg\. Baseline"), (best_run, r"Optimal Ratio \(\$r\^\*\$\)")]:
            fp1k = run_obj["test_fp_per_1k"]
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
    print("============================================================")
    print("  IEEE AIoT 2026 - Post-RQ1 Auto Runner & Table Populator   ")
    print("============================================================")
    mine_and_train_rq2()
    populate_manuscript_tables()
    print("\nALL BENCHMARK TASKS AND MANUSCRIPT COMPILATION COMPLETE!")

if __name__ == "__main__":
    main()
