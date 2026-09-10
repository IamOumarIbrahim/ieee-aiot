import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

img_path = "data/processed/RGB/images/subject_08/video_02/subject_08_video_02_frame_0207.jpg"
img = Image.open(img_path)

# Let's create a 2-panel figure: (a) 0% Baseline (Hallucination) vs (b) 80% Negative (Silence)
# We can also do a 3-panel if appropriate. Let's create both and inspect!

# Variation 1: 2-panel side by side
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), dpi=300)

# (a) Baseline 0%
axes[0].imshow(img)
# Draw bounding box for YOLO26n (phone_use 0.70) in red or orange, and YOLO11n (drinking 0.53) in cyan/yellow
# Or show YOLO26n [0, 226.5, 176.9, 529.6]
rect26 = patches.Rectangle((0, 226.5), 176.9, 529.6 - 226.5, linewidth=2.5, edgecolor='#E63946', facecolor='none')
axes[0].add_patch(rect26)
axes[0].text(8, 220, 'phone_use 0.70 (YOLO26n)', color='white', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.2', facecolor='#E63946', edgecolor='none', alpha=0.9))

rect11 = patches.Rectangle((0, 230.2), 192.6, 519.7 - 230.2, linewidth=2.0, linestyle='--', edgecolor='#F4A261', facecolor='none')
axes[0].add_patch(rect11)
axes[0].text(8, 510, 'drinking 0.53 (YOLO11n)', color='white', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.2', facecolor='#F4A261', edgecolor='none', alpha=0.9))

axes[0].set_title('(a) Baseline Detectors ($r = 0\%$)\nSpurious Foreground Hallucinations', fontsize=10, fontweight='bold', pad=8)
axes[0].axis('off')

# Status banner on bottom left of (a)
axes[0].text(320, 615, 'False Alarm Leakage', color='white', fontsize=9, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#D90429', edgecolor='white', linewidth=1.2, alpha=0.92))

# (b) Regularized 80%
axes[1].imshow(img)
axes[1].set_title('(b) Regularized Detectors ($r = 80\%$)\nClean Background Silence', fontsize=10, fontweight='bold', pad=8)
axes[1].axis('off')

# Status banner on bottom right of (b)
axes[1].text(320, 615, 'Zero False Alarms (Clean Silence)', color='white', fontsize=9, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#2A9D8F', edgecolor='white', linewidth=1.2, alpha=0.92))

plt.tight_layout()
plt.savefig("scratch/fig_qualitative_2panel.png", dpi=300, bbox_inches='tight')
plt.close()

# Variation 2: 3-panel side by side: (a) Baseline YOLO26n (Conv), (b) Baseline YOLO11n (Attention), (c) 80% Negatives (Silent)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=300)

# (a) YOLO26n 0%
axes[0].imshow(img)
r26 = patches.Rectangle((0, 226.5), 176.9, 529.6 - 226.5, linewidth=2.5, edgecolor='#E63946', facecolor='none')
axes[0].add_patch(r26)
axes[0].text(8, 220, 'phone_use 0.70', color='white', fontsize=8.5, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.2', facecolor='#E63946', edgecolor='none', alpha=0.9))
axes[0].set_title('(a) Pure Conv ($r{=}0\%$)\nYOLO26n Baseline', fontsize=9.5, fontweight='bold', pad=6)
axes[0].axis('off')
axes[0].text(320, 615, 'FP: Ambient Texture', color='white', fontsize=8.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#D90429', edgecolor='white', alpha=0.9))

# (b) YOLO11n 0%
axes[1].imshow(img)
r11 = patches.Rectangle((0, 230.2), 192.6, 519.7 - 230.2, linewidth=2.5, edgecolor='#E76F51', facecolor='none')
axes[1].add_patch(r11)
axes[1].text(8, 220, 'drinking 0.53', color='white', fontsize=8.5, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.2', facecolor='#E76F51', edgecolor='none', alpha=0.9))
axes[1].set_title('(b) Attention ($r{=}0\%$)\nYOLO11n Baseline', fontsize=9.5, fontweight='bold', pad=6)
axes[1].axis('off')
axes[1].text(320, 615, 'FP: Context Binding', color='white', fontsize=8.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#E76F51', edgecolor='white', alpha=0.9))

# (c) 80% Negatives
axes[2].imshow(img)
axes[2].set_title('(c) Regularized ($r{=}80\%$)\nArchitecture-Wide Silence', fontsize=9.5, fontweight='bold', pad=6)
axes[2].axis('off')
axes[2].text(320, 615, 'Zero False Alarms', color='white', fontsize=8.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#2A9D8F', edgecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig("scratch/fig_qualitative_3panel.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved 2panel and 3panel figures")
