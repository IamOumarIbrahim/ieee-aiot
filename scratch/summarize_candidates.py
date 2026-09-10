import cv2
import json

with open("scratch/candidates.json") as f:
    data = json.load(f)

print(f"Total candidates: {len(data)}")

# Let's save crops or summary of all 137 candidates with their image path and boxes
summary = []
for i, d in enumerate(data):
    b11 = d.get("yolo11_0", [])
    b26 = d.get("yolo26_0", [])
    summary.append({
        "id": i,
        "img": d["img"],
        "yolo11": b11,
        "yolo26": b26
    })

with open("scratch/cand_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Saved cand_summary.json")
