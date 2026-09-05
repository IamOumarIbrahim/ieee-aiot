"""
train_dfine_sweep.py - Automated D-FINE-N Ratio Sweep Launcher.

Orchestrates sequential training across the 5 negative-ratio configurations using the upstream D-FINE engine:
- Model: D-FINE-N (HGNetv2-B0 backbone)
- 5 arithmetic ratio splits: 0%, 20%, 40%, 60%, 80%
- Frozen RTX 4060 protocol: physical batch 4, accum steps 8 (effective batch 32), 160 epochs, stop_epoch 148, seed 42
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

DFINE_SPLITS = [
    {"name": "dfine_00_pos_only", "ratio": "0%", "config": "configs/dfine/dfine_00_pos_only.yml"},
    {"name": "dfine_20_low_neg", "ratio": "20%", "config": "configs/dfine/dfine_20_low_neg.yml"},
    {"name": "dfine_40_mod_neg", "ratio": "40%", "config": "configs/dfine/dfine_40_mod_neg.yml"},
    {"name": "dfine_60_high_neg", "ratio": "60%", "config": "configs/dfine/dfine_60_high_neg.yml"},
    {"name": "dfine_80_max_neg", "ratio": "80%", "config": "configs/dfine/dfine_80_max_neg.yml"},
]

def parse_args():
    parser = argparse.ArgumentParser(description="Automated D-FINE Ratio Sweep Launcher")
    parser.add_argument("--dfine-dir", type=str, default="DFINE", help="Path to cloned upstream Peterande/D-FINE repository")
    parser.add_argument("--splits", type=str, default="all", help="Comma-separated split keys (e.g., '00,20' or 'all')")
    parser.add_argument("--device", type=str, default="0", help="CUDA visible device index")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration paths and D-FINE entrypoint")
    return parser.parse_args()

def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    dfine_path = Path(args.dfine_dir).resolve()
    train_py = dfine_path / "train.py"

    selected_splits = DFINE_SPLITS
    if args.splits != "all":
        filter_keys = [k.strip() for k in args.splits.split(",")]
        selected_splits = [s for s in DFINE_SPLITS if any(k in s["name"] for k in filter_keys)]

    print("============================================================")
    print("  D-FINE-N Negative-Frame Ratio Sweep Launcher")
    print(f"  Upstream D-FINE directory: {dfine_path}")
    print(f"  Target splits: {[s['name'] for s in selected_splits]}")
    print("  Protocol: batch=4, accum=8 (eff=32), epochs=160, stop_epoch=148, seed=42")
    print("============================================================\n")

    if args.dry_run:
        print("[DRY-RUN] Checking D-FINE installation and config paths:")
        print(f"  - D-FINE engine train.py exists: {train_py.exists()} ({train_py})")
        import yaml

        def deep_merge(base, update):
            merged = dict(base)
            for k, v in update.items():
                if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = deep_merge(merged[k], v)
                else:
                    merged[k] = v
            return merged

        def load_yaml_with_includes(file_path):
            file_path = Path(file_path).resolve()
            with open(file_path, "r") as f:
                data = yaml.safe_load(f) or {}
            merged = {}
            if "__include__" in data:
                includes = data["__include__"]
                if isinstance(includes, str):
                    includes = [includes]
                for inc in includes:
                    inc_path = (file_path.parent / inc).resolve()
                    base_data = load_yaml_with_includes(inc_path)
                    merged = deep_merge(merged, base_data)
            merged = deep_merge(merged, data)
            return merged

        for s in selected_splits:
            cfg = repo_root / s["config"]
            exists = cfg.exists()
            status = "FOUND" if exists else "MISSING"
            print(f"  - {s['name']}: [{status}] ({cfg})")
            if exists:
                try:
                    data = load_yaml_with_includes(cfg)
                    num_cls = data.get("num_classes")
                    ann = data.get("train_dataloader", {}).get("dataset", {}).get("ann_file")
                    ann_exists = (repo_root / ann).exists() if ann else False
                    print(f"      Verified: classes={num_cls}, ann_file={ann} (exists={ann_exists})")
                except Exception as e:
                    print(f"      [ERROR loading YAML]: {e}")
        print("\nDry run completed.")
        return

    if not train_py.exists():
        print(f"[ERROR] D-FINE train.py not found at: {train_py}")
        print("Please clone Peterande/D-FINE (e.g. `git clone https://github.com/Peterande/D-FINE.git DFINE`) or specify --dfine-dir.")
        sys.exit(1)

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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device

    for idx, s in enumerate(selected_splits, 1):
        cfg_abs = (repo_root / s["config"]).resolve()
        print(f"\n[{idx}/{len(selected_splits)}] Launching D-FINE-N: {s['name']} ({s['ratio']} Negatives)")

        cmd = [
            sys.executable,
            str(train_py),
            "-c", str(cfg_abs),
            "--use-amp",
            "--seed", "42"
        ]

        print(f"  Command: {' '.join(cmd)}")
        ret = subprocess.run(cmd, cwd=str(dfine_path), env=env)
        if ret.returncode != 0:
            print(f"[ERROR] Run {s['name']} failed with returncode {ret.returncode}")
            sys.exit(ret.returncode)
        else:
            print(f"  [OK] Run {s['name']} finished successfully.")

if __name__ == "__main__":
    main()
