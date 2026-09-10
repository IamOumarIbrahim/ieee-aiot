import cv2
import numpy as np
import json
from pathlib import Path

def get_entropy_and_laplacian(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Shannon entropy of pixel intensity distribution
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    shannon_entropy = -np.sum(hist * np.log2(hist))
    
    # Laplacian variance (high-frequency edge/texture density)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return float(shannon_entropy), float(lap_var)

# Check some random negatives
with open("data/processed/RGB/coco/instances_train_00_pos_only.json") as f:
    pos_ids = {img['id'] for img in json.load(f)['images']}
with open("data/processed/RGB/coco/instances_train_40_mod_neg.json") as f:
    r40_imgs = json.load(f)['images']
    r40_negs = [img for img in r40_imgs if img['id'] not in pos_ids]

# Check candidate negatives
with open("scratch/candidates.json") as f:
    cand_data = json.load(f)

print("Scoring random negatives...")
r_scores = []
for item in r40_negs[:100]:
    p = Path("data/processed/RGB") / item["file_name"]
    ent, lap = get_entropy_and_laplacian(p)
    r_scores.append((ent, lap, str(p)))

r_scores.sort(key=lambda x: x[0])
print("\nLowest entropy random negatives:")
for ent, lap, p in r_scores[:10]:
    print(f"Entropy: {ent:.3f}, LapVar: {lap:.1f} -> {p}")

print("\nScoring hard-mined candidates...")
c_scores = []
for item in cand_data:
    p = Path(item["img"])
    ent, lap = get_entropy_and_laplacian(p)
    c_scores.append((ent, lap, str(p), item.get('yolo11_0', []), item.get('yolo26_0', [])))

c_scores.sort(key=lambda x: x[0], reverse=True)
print("\nHighest entropy hard candidates:")
for ent, lap, p, y11, y26 in c_scores[:10]:
    print(f"Entropy: {ent:.3f}, LapVar: {lap:.1f} -> {p}")
    print(f"   y11={y11}, y26={y26}")
