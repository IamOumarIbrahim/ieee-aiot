import yaml
import glob
from pathlib import Path

repo_root = Path(r"C:\Dev\repos\Public repos\ieee-aiot")
args_files = sorted(glob.glob(str(repo_root / "runs" / "*" / "*" / "args.yaml")))

print(f"Found {len(args_files)} args.yaml files:")
common_keys = None
all_cfgs = {}

for f in args_files:
    rel = Path(f).relative_to(repo_root)
    with open(f, "r") as stream:
        cfg = yaml.safe_load(stream)
    all_cfgs[str(rel)] = cfg

# Check variations across all runs
keys = set()
for c in all_cfgs.values():
    keys.update(c.keys())

variations = {}
first_run = list(all_cfgs.keys())[0]
first_cfg = all_cfgs[first_run]

for k in sorted(keys):
    vals = {r: all_cfgs[r].get(k) for r in all_cfgs}
    unique_vals = set(str(v) for v in vals.values())
    if len(unique_vals) > 1:
        variations[k] = vals

print(f"\nHyperparameters that vary across runs ({len(variations)} keys):")
for k, v in variations.items():
    print(f"  Key '{k}':")
    # print summary of unique values
    unique_map = {}
    for run_name, val in v.items():
        unique_map.setdefault(str(val), []).append(run_name.split('\\')[0] + '/' + run_name.split('\\')[1])
    for val_str, runs in unique_map.items():
        print(f"    Value: {val_str} -> {len(runs)} runs (e.g. {runs[:2]})")

print("\n--- Hyperparameters that are IDENTICAL across ALL runs: ---")
for k in sorted(keys):
    if k not in variations:
        print(f"  {k}: {first_cfg.get(k)}")
