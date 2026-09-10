import json
print('=== MULTI SEED RQ1 SUMMARY ===')
with open('runs/multi_seed_rq1_summary.json') as f:
    d = json.load(f)
for m, splits in d.items():
    print(f'Model: {m}')
    for s in splits:
        print(f"  {s['split']} ({s['ratio']}) -> mAP50: {s['map50_mean']} +- {s['map50_std']}, mAP50:95: {s['map50_95_mean']} +- {s['map50_95_std']}, P: {s['precision_mean']}, R: {s['recall_mean']}, FP/1k: {s['fp_per_1k_mean']} +- {s['fp_per_1k_std']}, raw_FPs: {s['raw_fps_avg']}")
