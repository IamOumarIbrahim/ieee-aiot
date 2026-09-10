import json
import cv2
import numpy as np
from pathlib import Path

with open("data/annotations/RGB/annotations.json") as f:
    master = json.load(f)

img_to_anns = {img["id"]: [] for img in master["images"]}
for ann in master["annotations"]:
    img_to_anns[ann["image_id"]].append(ann)

negs = [img for img in master["images"] if len(img_to_anns[img["id"]]) == 0]
print(f"Total negatives in dataset: {len(negs)}")

# Group negatives by subject and video
grouped = {}
for img in negs:
    fn = img["file_name"]
    parts = fn.split("/")
    subj = parts[1]
    vid = parts[2]
    grouped.setdefault((subj, vid), []).append(img)

print("Negative counts by subject/video:")
for k, v in list(grouped.items())[:15]:
    print(f"  {k}: {len(v)} negatives")
