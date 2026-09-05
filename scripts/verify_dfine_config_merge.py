"""
verify_dfine_config_merge.py - D-FINE Config Include-Chain Precedence & Hyperparameter Verification.

Pure-Python + PyYAML verification script:
- Zero torch/CUDA imports (100% GPU-free, safe to run during active training).
- Walks the __include__ chain replicating upstream D-FINE's exact merge/override precedence
  as implemented in DFINE/src/core/yaml_utils.py.
- Inspects and prints resolved values for:
  * epochs
  * lr (optimizer base learning rate)
  * backbone lr (extracted from optimizer param-groups matching backbone regex)
  * weight_decay
  * total_batch_size (train_dataloader)
  * num_classes
  * stop_epoch (collate_fn & transforms policy)
  * train annotation file
  * output_dir
"""

import os
import sys
import copy
import re
from pathlib import Path
import yaml

INCLUDE_KEY = "__include__"


def merge_dict(dct: dict, another_dct: dict) -> dict:
    """
    Replicates D-FINE's exact merge_dict from DFINE/src/core/yaml_utils.py:
    Recursively merges sub-dictionaries; replaces scalar or list entries with the update.
    """
    for k in another_dct:
        if k in dct and isinstance(dct[k], dict) and isinstance(another_dct[k], dict):
            merge_dict(dct[k], another_dct[k])
        else:
            dct[k] = copy.deepcopy(another_dct[k])
    return dct


def load_config_with_trace(file_path: Path | str, trace: list = None) -> tuple[dict, list]:
    """
    Loads YAML config and recursively resolves __include__ directives adhering to D-FINE precedence.
    Returns (resolved_dict, trace_list).
    """
    if trace is None:
        trace = []

    file_path = Path(file_path).resolve()
    trace.append(str(file_path))

    with open(file_path, "r", encoding="utf-8") as f:
        file_cfg = yaml.load(f, Loader=yaml.SafeLoader) or {}

    cfg = {}
    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)

            if not os.path.isabs(base_yaml):
                base_yaml = os.path.normpath(file_path.parent / base_yaml)

            base_cfg, _ = load_config_with_trace(base_yaml, trace)
            merge_dict(cfg, base_cfg)

    merge_dict(cfg, file_cfg)
    return cfg, trace


def extract_backbone_lr(optimizer_cfg: dict) -> tuple[float | None, str | None]:
    """
    Finds the learning rate assigned to backbone parameters in the optimizer param groups.
    """
    params = optimizer_cfg.get("params", [])
    if not isinstance(params, list):
        return None, None

    for pg in params:
        pattern = pg.get("params", "")
        if "backbone" in pattern and "lr" in pg:
            return pg["lr"], pattern
    return None, None


def extract_stop_epoch(cfg: dict) -> int | None:
    """
    Extracts stop_epoch from collate_fn or transforms policy.
    """
    collate = cfg.get("train_dataloader", {}).get("collate_fn", {})
    if isinstance(collate, dict) and "stop_epoch" in collate:
        return collate["stop_epoch"]

    transforms = cfg.get("train_dataloader", {}).get("dataset", {}).get("transforms", {})
    if isinstance(transforms, dict):
        policy = transforms.get("policy", {})
        if isinstance(policy, dict) and "epoch" in policy:
            return policy["epoch"]

    return None


def main():
    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs" / "dfine"

    ratio_configs = [
        ("dfine_00_pos_only", "0%", configs_dir / "dfine_00_pos_only.yml"),
        ("dfine_20_low_neg", "20%", configs_dir / "dfine_20_low_neg.yml"),
        ("dfine_40_mod_neg", "40%", configs_dir / "dfine_40_mod_neg.yml"),
        ("dfine_60_high_neg", "60%", configs_dir / "dfine_60_high_neg.yml"),
        ("dfine_80_max_neg", "80%", configs_dir / "dfine_80_max_neg.yml"),
    ]

    print("=" * 85)
    print("  D-FINE CONFIGURATION INCLUDE-CHAIN & RESOLVED HYPERPARAMETERS")
    print("  Precedence Rule: Later includes override earlier; child overrides all includes.")
    print("=" * 85)

    summary_rows = []

    for name, ratio, cfg_path in ratio_configs:
        print(f"\n--- Checking {name} ({ratio} Negatives) ---")
        if not cfg_path.exists():
            print(f"  [ERROR] File not found: {cfg_path}")
            continue

        resolved_cfg, trace = load_config_with_trace(cfg_path, trace=[])

        print(f"  Config Path: {cfg_path}")
        print(f"  Include Chain Traversed ({len(trace)} files):")
        for i, step in enumerate(trace, 1):
            rel_step = os.path.relpath(step, repo_root)
            print(f"    [{i}] {rel_step}")

        # Extract resolved hyperparameters
        epochs = resolved_cfg.get("epochs")
        optimizer = resolved_cfg.get("optimizer", {})
        base_lr = optimizer.get("lr")
        backbone_lr, pattern = extract_backbone_lr(optimizer)
        weight_decay = optimizer.get("weight_decay")
        train_loader = resolved_cfg.get("train_dataloader", {})
        total_batch_size = train_loader.get("total_batch_size")
        num_classes = resolved_cfg.get("num_classes")
        stop_epoch = extract_stop_epoch(resolved_cfg)
        ann_file = train_loader.get("dataset", {}).get("ann_file")
        ann_exists = (repo_root / ann_file).exists() if ann_file else False
        output_dir = resolved_cfg.get("output_dir")

        print(f"\n  Resolved Values for {name}:")
        print(f"    - epochs:           {epochs}")
        print(f"    - num_classes:      {num_classes}")
        print(f"    - base lr:          {base_lr}")
        print(f"    - backbone lr:      {backbone_lr} (from pattern: '{pattern}')")
        print(f"    - weight_decay:     {weight_decay}")
        print(f"    - total_batch_size: {total_batch_size}")
        print(f"    - stop_epoch:       {stop_epoch}")
        print(f"    - output_dir:       {output_dir}")
        print(f"    - train ann_file:   {ann_file} (exists: {ann_exists})")

        summary_rows.append({
            "name": name,
            "ratio": ratio,
            "epochs": epochs,
            "num_classes": num_classes,
            "base_lr": base_lr,
            "backbone_lr": backbone_lr,
            "weight_decay": weight_decay,
            "total_batch_size": total_batch_size,
            "stop_epoch": stop_epoch,
            "ann_exists": ann_exists,
        })

    print("\n" + "=" * 105)
    print("  CONSOLIDATED RESOLVED HYPERPARAMETER SUMMARY TABLE")
    print("=" * 105)
    header = f"{'Config':<20} | {'Ratio':<6} | {'Epochs':<6} | {'Classes':<7} | {'Base LR':<8} | {'Bkbn LR':<8} | {'W-Decay':<8} | {'Batch':<5} | {'Stop-Ep':<7} | {'Ann Exists':<10}"
    print(header)
    print("-" * 105)
    for r in summary_rows:
        print(f"{r['name']:<20} | {r['ratio']:<6} | {r['epochs']:<6} | {r['num_classes']:<7} | {r['base_lr']:<8} | {r['backbone_lr']:<8} | {r['weight_decay']:<8} | {r['total_batch_size']:<5} | {r['stop_epoch']:<7} | {str(r['ann_exists']):<10}")
    print("=" * 105)

    # Sanity checks
    all_epochs_160 = all(r["epochs"] == 160 for r in summary_rows)
    all_classes_4 = all(r["num_classes"] == 4 for r in summary_rows)
    all_batch_4 = all(r["total_batch_size"] == 4 for r in summary_rows)
    all_stop_148 = all(r["stop_epoch"] == 148 for r in summary_rows)
    all_anns_exist = all(r["ann_exists"] for r in summary_rows)

    print("\nPrecedence Verification Checkpoints:")
    print(f"  [OK] All configs resolve to epochs = 160:          {all_epochs_160}")
    print(f"  [OK] All configs resolve to num_classes = 4:       {all_classes_4}")
    print(f"  [OK] All configs resolve to total_batch_size = 4:  {all_batch_4}")
    print(f"  [OK] All configs resolve to stop_epoch = 148:      {all_stop_148}")
    print(f"  [OK] All train annotation files exist on disk:     {all_anns_exist}")

    if all([all_epochs_160, all_classes_4, all_batch_4, all_stop_148, all_anns_exist]):
        print("\n>>> ALL D-FINE MERGE OVERRIDES VERIFIED CORRECT AND READY FOR TRAINING <<<")
    else:
        print("\n>>> WARNING: One or more configurations failed hyperparameter verification <<<")


if __name__ == "__main__":
    main()
