import json
import cv2
import numpy as np
from pathlib import Path

pool_path = Path("data/processed/RGB/train_neg_pool_ids.json")
with open(pool_path) as f:
    pool_ids = set(json.load(f)["train_neg_pool_ids"])

master_path = Path("data/annotations/RGB/annotations.json")
with open(master_path) as f:
    master = json.load(f)

pool_images = [img for img in master["images"] if img["id"] in pool_ids]
print(f"Total pool negatives: {len(pool_images)}")

# Sample 500 images across pool and compute entropy and shadow contrast
results = []
step = len(pool_images) // 300
for i in range(0, len(pool_images), step):
    img_meta = pool_images[i]
    p = Path("data/processed/RGB") / img_meta["file_name"]
    img = cv2.imread(str(p))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Check shadow/lighting complexity
    # Gradient energy
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2).mean()
    
    results.append({
        'path': str(p),
        'entropy': float(entropy),
        'grad_mag': float(grad_mag),
        'std': float(gray.std()),
        'mean': float(gray.mean())
    })

# Lowest entropy (flat, uniform, featureless)
results.sort(key=lambda x: x['entropy'])
print("\n--- LOWEST ENTROPY NEGATIVES (Random candidates: flat, featureless) ---")
for r in results[:8]:
    print(f"Entropy: {r['entropy']:.3f}, GradMag: {r['grad_mag']:.1f}, Std: {r['std']:.1f} -> {r['path']}")

# Highest entropy (complex, textured, high information density)
results.sort(key=lambda x: x['entropy'], reverse=True)
print("\n--- HIGHEST ENTROPY NEGATIVES (Hard candidates: complex shadows/silhouettes) ---")
for r in results[:8]:
    print(f"Entropy: {r['entropy']:.3f}, GradMag: {r['grad_mag']:.1f}, Std: {r['std']:.1f} -> {r['path']}")
