import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

img_path = "data/processed/RGB/images/subject_08/video_02/subject_08_video_02_frame_0207.jpg"
img = Image.open(img_path)

out_dir = "docs/manuscript"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure A: 3-Panel Side-by-Side (Ideal for figure* double column or wide)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), dpi=300)

# Panel (a): YOLO26n 0% (Pure Conv)
axes[0].imshow(img)
r26 = patches.Rectangle((0, 226.5), 176.9, 529.6 - 226.5, linewidth=2.2, edgecolor='#D90429', facecolor='none')
axes[0].add_patch(r26)
axes[0].text(8, 216, 'phone_use 0.70', color='white', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.2', facecolor='#D90429', edgecolor='none', alpha=0.95))
axes[0].set_title(r'(a) Pure Conv ($r{=}0\%$, YOLO26n)' + '\nSpurious Texture Hallucination', fontsize=8.5, fontweight='bold', pad=5)
axes[0].axis('off')
axes[0].text(320, 615, 'False Alarm (Leakage)', color='white', fontsize=7.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#D90429', edgecolor='white', linewidth=0.8, alpha=0.95))

# Panel (b): YOLO11n 0% (Attention / Hybrid)
axes[1].imshow(img)
r11 = patches.Rectangle((0, 230.2), 192.6, 519.7 - 230.2, linewidth=2.2, edgecolor='#E76F51', facecolor='none')
axes[1].add_patch(r11)
axes[1].text(8, 216, 'drinking 0.53', color='white', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.2', facecolor='#E76F51', edgecolor='none', alpha=0.95))
axes[1].set_title(r'(b) Attention ($r{=}0\%$, YOLO11n)' + '\nSpurious Context Binding', fontsize=8.5, fontweight='bold', pad=5)
axes[1].axis('off')
axes[1].text(320, 615, 'False Alarm (Leakage)', color='white', fontsize=7.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#E76F51', edgecolor='white', linewidth=0.8, alpha=0.95))

# Panel (c): 80% Regularized (Silence)
axes[2].imshow(img)
axes[2].set_title(r'(c) Regularized ($r{=}80\%$, All Models)' + '\nClean Background Silence', fontsize=8.5, fontweight='bold', pad=5)
axes[2].axis('off')
axes[2].text(320, 615, 'Zero False Alarms (Clean)', color='white', fontsize=7.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#2A9D8F', edgecolor='white', linewidth=0.8, alpha=0.95))

plt.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.86, bottom=0.02)
fig.savefig(os.path.join(out_dir, "fig_qualitative_suppression.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, "fig_qualitative_suppression.pdf"), bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure B: 2-Panel Side-by-Side (Ideal for single-column \columnwidth)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(3.5, 1.95), dpi=300)

axes2[0].imshow(img)
r26_b = patches.Rectangle((0, 226.5), 176.9, 529.6 - 226.5, linewidth=1.8, edgecolor='#D90429', facecolor='none')
axes2[0].add_patch(r26_b)
axes2[0].text(6, 218, 'phone_use 0.70 (Y26)', color='white', fontsize=5.5, fontweight='bold',
              bbox=dict(boxstyle='square,pad=0.15', facecolor='#D90429', edgecolor='none', alpha=0.95))
r11_b = patches.Rectangle((0, 230.2), 192.6, 519.7 - 230.2, linewidth=1.5, linestyle='--', edgecolor='#F4A261', facecolor='none')
axes2[0].add_patch(r11_b)
axes2[0].text(6, 512, 'drinking 0.53 (Y11)', color='white', fontsize=5.5, fontweight='bold',
              bbox=dict(boxstyle='square,pad=0.15', facecolor='#F4A261', edgecolor='none', alpha=0.95))
axes2[0].set_title(r'(a) Baseline ($r{=}0\%$)', fontsize=7.5, fontweight='bold', pad=3)
axes2[0].axis('off')
axes2[0].text(320, 615, 'False Alarms', color='white', fontsize=5.5, fontweight='bold', ha='center',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#D90429', edgecolor='white', linewidth=0.6, alpha=0.95))

axes2[1].imshow(img)
axes2[1].set_title(r'(b) Regularized ($r{=}80\%$)', fontsize=7.5, fontweight='bold', pad=3)
axes2[1].axis('off')
axes2[1].text(320, 615, 'Clean Silence', color='white', fontsize=5.5, fontweight='bold', ha='center',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#2A9D8F', edgecolor='white', linewidth=0.6, alpha=0.95))

plt.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.86, bottom=0.02)
fig2.savefig(os.path.join(out_dir, "fig_qualitative_suppression_col.png"), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(out_dir, "fig_qualitative_suppression_col.pdf"), bbox_inches='tight')
plt.close(fig2)

print("Saved publication figures successfully in docs/manuscript!")
