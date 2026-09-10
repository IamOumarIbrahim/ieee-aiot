import json

with open('scratch/candidates.json') as f:
    data = json.load(f)

print(f"Total candidates: {len(data)}")
both_count = 0
for i, d in enumerate(data):
    has_11 = len(d['yolo11_0']) > 0
    has_26 = len(d['yolo26_0']) > 0
    if has_11 and has_26:
        both_count += 1
        print(f"[{i}] BOTH: {d['img']}")
        print(f"    YOLO11_0: {d['yolo11_0']}")
        print(f"    YOLO26_0: {d['yolo26_0']}")

print(f"\nBoth hallucinated in {both_count} frames.")
