import json
import cv2
import os

with open('scratch/candidates.json') as f:
    data = json.load(f)

# Let's inspect 10 frames that have detections
for idx in [41, 57, 60, 70, 90, 91, 119]:
    d = data[idx]
    print(f"Candidate {idx}: {d['img']}")
    print(f"  YOLO11: {d['yolo11_0']}")
    print(f"  YOLO26: {d['yolo26_0']}")

