import json
import cv2
import numpy as np
from pathlib import Path

# Load splits
with open("data/processed/RGB/coco/instances_train_00_pos_only.json") as f:
    pos_ids = {img['id'] for img in json.load(f)['images']}

with open("data/processed/RGB/coco/instances_train_40_mod_neg.json") as f:
    r40_imgs = json.load(f)['images']
    r40_negs = [img for img in r40_imgs if img['id'] not in pos_ids]

with open("data/processed/RGB/coco/instances_train_curated_yolo11n_best_curated.json") as f:
    c11_imgs = json.load(f)['images']
    c11_negs = [img for img in c11_imgs if img['id'] not in pos_ids]

print(f"Random 40% negatives: {len(r40_negs)}")
print(f"Curated 11n negatives: {len(c11_negs)}")

def compute_metrics(neg_list, sample_size=400):
    entropies = []
    grad_mags = []
    lap_vars = []
    
    # Step through uniformly
    step = max(1, len(neg_list) // sample_size)
    sampled = neg_list[::step][:sample_size]
    
    for img_meta in sampled:
        p = Path("data/processed/RGB") / img_meta["file_name"]
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        h = -np.sum(hist * np.log2(hist))
        
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gm = np.sqrt(gx**2 + gy**2).mean()
        
        lv = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        entropies.append(h)
        grad_mags.append(gm)
        lap_vars.append(lv)
        
    return np.array(entropies), np.array(grad_mags), np.array(lap_vars)

r_ent, r_grad, r_lap = compute_metrics(r40_negs, sample_size=500)
c_ent, c_grad, c_lap = compute_metrics(c11_negs, sample_size=500)

print("\n--- STATISTICAL COMPARISON (RQ2 FOUNDATION) ---")
print(f"Random Negatives Entropy:    mean={r_ent.mean():.3f} +/- {r_ent.std():.3f}, median={np.median(r_ent):.3f}")
print(f"Curated Negatives Entropy:   mean={c_ent.mean():.3f} +/- {c_ent.std():.3f}, median={np.median(c_ent):.3f}")
print(f"Random Negatives Grad Mag:   mean={r_grad.mean():.1f} +/- {r_grad.std():.1f}")
print(f"Curated Negatives Grad Mag:  mean={c_grad.mean():.1f} +/- {c_grad.std():.1f}")
print(f"Random Negatives Lap Var:    mean={r_lap.mean():.1f} +/- {r_lap.std():.1f}")
print(f"Curated Negatives Lap Var:   mean={c_lap.mean():.1f} +/- {c_lap.std():.1f}")
