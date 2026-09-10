import json
import os
import cv2
import numpy as np

with open("scratch/candidates.json") as f:
    cand_data = json.load(f)

print(f"Total candidates in candidates.json: {len(cand_data)}")

# Let's see some candidates with high confidence detections
print("\nTop candidates by max confidence:")
for i, d in enumerate(cand_data[:15]):
    y11 = d.get('yolo11_0', [])
    y26 = d.get('yolo26_0', [])
    print(f"Cand {i}: img={d['img']}")
    if y11:
        print(f"   yolo11: {y11}")
    if y26:
        print(f"   yolo26: {y26}")

# Also inspect random negatives from 40% random split
with open("data/processed/RGB/coco/instances_train_00_pos_only.json") as f:
    pos_ids = {img['id'] for img in json.load(f)['images']}
with open("data/processed/RGB/coco/instances_train_40_mod_neg.json") as f:
    r40_imgs = json.load(f)['images']
    r40_negs = [img for img in r40_imgs if img['id'] not in pos_ids]

print(f"\nRandom negatives count: {len(r40_negs)}")
print("Sample random negatives:")
for img in r40_negs[:10]:
    print(f"  id={img['id']}, file_name={img['file_name']}")
