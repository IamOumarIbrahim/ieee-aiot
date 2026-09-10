import json

with open("scratch/candidates.json") as f:
    data = json.load(f)

print(f"Total candidates: {len(data)}")

# Let us filter for detections where bounding box is in the lower region or steering wheel area or background
steering_wheel_cands = []
for idx, d in enumerate(data):
    # Check boxes
    all_boxes = d["yolo11_0"] + d["yolo26_0"]
    for lbl, conf, box in all_boxes:
        x1, y1, x2, y2 = box
        # Steering wheel is usually y1 > 350 or x around center/bottom, or visor/windshield
        # Let us print any candidates with boxes in lower half (y1 > 300) or high conf
        if y2 > 450 and (y2 - y1) < 200: # wide or localized cabin feature
            steering_wheel_cands.append((idx, d["img"], lbl, conf, box))

print(f"Found {len(steering_wheel_cands)} potential steering/cabin feature detections:")
for c in steering_wheel_cands[:20]:
    print(c)

