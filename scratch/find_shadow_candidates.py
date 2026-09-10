import json
import cv2
import numpy as np
from pathlib import Path

with open("scratch/candidates.json") as f:
    cand_data = json.load(f)

# Let's check all 137 candidates for visual characteristics:
# 1. High contrast / shadow / dynamic range (std dev of brightness)
# 2. Strong gradients (Sobel / Laplacian)
# 3. Model detections (labels, confidences, boxes)

results = []
for i, d in enumerate(cand_data):
    img_path = d["img"]
    img = cv2.imread(img_path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Measure contrast and dark/bright shadow areas
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    
    # Shadow ratio (fraction of very dark pixels < 40)
    shadow_ratio = np.mean(gray < 40)
    # Highlight ratio (fraction of bright pixels > 220)
    highlight_ratio = np.mean(gray > 220)
    
    y11 = d.get('yolo11_0', [])
    y26 = d.get('yolo26_0', [])
    
    results.append({
        'id': i,
        'path': img_path,
        'entropy': entropy,
        'lap_var': lap_var,
        'contrast': contrast,
        'shadow_ratio': shadow_ratio,
        'highlight_ratio': highlight_ratio,
        'y11': y11,
        'y26': y26
    })

# Sort by contrast / shadow-highlight dynamic range
results.sort(key=lambda x: (x['shadow_ratio'] + x['highlight_ratio']), reverse=True)
print("Top candidates with extreme lighting/shadows/contrast:")
for r in results[:10]:
    print(f"Cand {r['id']}: {r['path']}")
    print(f"  Entropy={r['entropy']:.2f}, Contrast={r['contrast']:.1f}, Shadows={r['shadow_ratio']:.2f}, Highlights={r['highlight_ratio']:.2f}")
    print(f"  y11={r['y11']}")
    print(f"  y26={r['y26']}")
