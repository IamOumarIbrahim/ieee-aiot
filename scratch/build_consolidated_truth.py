import json
import glob
import os
from pathlib import Path

repo_root = Path(r"C:\Dev\repos\Public repos\ieee-aiot")

# Gather all data
full_data = {}

# 1. Split stats
with open(repo_root / "data" / "processed" / "RGB" / "split_stats.json") as f:
    full_data["split_stats"] = json.load(f)

# 2. Curation stats
curation_stats = {}
for p in sorted(glob.glob(str(repo_root / "runs" / "curation_stats" / "*.json"))):
    fn = Path(p).name
    with open(p) as f:
        curation_stats[fn] = json.load(f)
full_data["curation_stats"] = curation_stats

# 3. Model sweep summaries
sweeps = {}
for p in sorted(glob.glob(str(repo_root / "runs" / "*sweep*" / "*sweep_summary.json"))):
    folder = Path(p).parent.name
    fn = Path(p).name
    with open(p) as f:
        sweeps[folder] = json.load(f)
full_data["sweeps"] = sweeps

# 4. RQ2 curation summaries
rq2_summaries = {}
for p in [repo_root / "runs" / "rq2_curation_summary.json", repo_root / "runs" / "rq2_curation_summary_seed43.json"]:
    if p.exists():
        with open(p) as f:
            rq2_summaries[p.name] = json.load(f)
full_data["rq2_summaries"] = rq2_summaries

# 5. Multi-seed summaries
for fn in ["multi_seed_rq1_summary.json", "multi_seed_rq2_summary.json", "multi_seed_rq2_summary_3seeds.json"]:
    p = repo_root / "runs" / fn
    if p.exists():
        with open(p) as f:
            full_data[fn] = json.load(f)

# Dump consolidated data
with open(repo_root / "scratch" / "consolidated_truth.json", "w", encoding="utf-8") as f:
    json.dump(full_data, f, indent=2)

print("Consolidated all ground-truth JSON files successfully!")
print("Top-level keys in consolidated_truth.json:", list(full_data.keys()))
