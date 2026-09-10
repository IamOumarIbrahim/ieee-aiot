import json
import glob
import os
from pathlib import Path

repo_root = Path(r"C:\Dev\repos\Public repos\ieee-aiot")

print("=== ALL SUMMARY JSON FILES ===")
summary_files = sorted(glob.glob(str(repo_root / "runs" / "*.json")) + glob.glob(str(repo_root / "runs" / "*" / "*.json")))

for p in summary_files:
    rel_p = os.path.relpath(p, repo_root)
    print("=" * 90)
    print(f"FILE: {rel_p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "protocol" in data:
        print(f"Model: {data.get('model')}, Timestamp: {data.get('timestamp')}")
        print(f"Protocol: {data.get('protocol')}")
    
    if "runs" in data:
        print(f"Runs count: {len(data['runs'])}")
        for r in data["runs"]:
            sp = r.get("split")
            ratio = r.get("ratio")
            v_m50 = r.get("val_map50")
            v_m = r.get("val_map50_95")
            v_fp = r.get("val_fp_per_1k")
            t_m50 = r.get("test_map50")
            t_m = r.get("test_map50_95")
            t_fp = r.get("test_fp_per_1k")
            t_fps = r.get("total_test_fps")
            t_neg = r.get("test_neg_frames")
            t_time = r.get("train_time_minutes")
            print(f"  Split: {sp:<35} | Ratio: {ratio:<5} | Val mAP50: {v_m50} | Val FP/1k: {v_fp} | Test mAP50: {t_m50} | Test mAP50:95: {t_m} | Test FP/1k: {t_fp} | FPs: {t_fps}/{t_neg} | Time(m): {t_time}")
    else:
        if isinstance(data, dict):
            print(f"Dict keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, dict):
                    if "random" in v and "curated" in v:
                        rnd = v["random"]
                        cur = v["curated"]
                        print(f"    Key {k} -> Random:  mAP50={rnd.get('test_map50')}, mAP50:95={rnd.get('test_map50_95')}, FP/1k={rnd.get('test_fp_per_1k')}")
                        print(f"           -> Curated: mAP50={cur.get('test_map50')}, mAP50:95={cur.get('test_map50_95')}, FP/1k={cur.get('test_fp_per_1k')}")
                        print(f"           -> Delta FP/1k: {v.get('delta_fp_per_1k')}")
                    elif "random" in v and "curated_3seeds" in v:
                        rnd = v["random"]
                        cur = v["curated_3seeds"]
                        print(f"    Key {k} -> Random:  mAP50_mean={rnd.get('map50_mean')}+-{rnd.get('map50_std')}, FP/1k={rnd.get('fp_per_1k_mean')}+-{rnd.get('fp_per_1k_std')}")
                        print(f"           -> Curated: mAP50_mean={cur.get('map50_mean')}+-{cur.get('map50_std')}, FP/1k={cur.get('fp_per_1k_mean')}+-{cur.get('fp_per_1k_std')}")
                        print(f"           -> Pct Delta: {v.get('pct_delta')}%, Delta: {v.get('delta_fp_per_1k')}")
                    else:
                        print(f"    Key {k}: {v}")
                elif isinstance(v, list):
                    print(f"  Key '{k}': list with {len(v)} items")
                    for item in v:
                        if isinstance(item, dict) and "split" in item:
                            print(f"    {item.get('split')}: map50_mean={item.get('map50_mean')}+-{item.get('map50_std')}, fp_mean={item.get('fp_per_1k_mean')}+-{item.get('fp_per_1k_std')}")
