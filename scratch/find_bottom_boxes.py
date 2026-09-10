import json

with open("scratch/candidates.json") as f:
    data = json.load(f)

print(f"Total candidates: {len(data)}")
for i, d in enumerate(data):
    for lbl, conf, box in d.get("yolo11_0", []) + d.get("yolo26_0", []):
        x1, y1, x2, y2 = box
        # check bottom area (steering wheel / dashboard)
        if y1 > 350 and y2 > 500:
            print(f"[{i}] {d['img']}: {lbl} {conf:.2f} box={box}")

