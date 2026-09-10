import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Clean, modern publication aesthetics matching IEEE Transactions style
plt.rcParams['font.family'] = 'DejaVu Sans'

img_random_path = "data/processed/RGB/images/subject_01/video_01/subject_01_video_01_frame_0004.jpg"
img_hard_path = "data/processed/RGB/images/subject_01/video_02/subject_01_video_02_frame_0210.jpg"

img_rand = Image.open(img_random_path)
img_hard = Image.open(img_hard_path)

out_dir = "docs/manuscript"
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 1. Double-Column 3-Panel Version (fig_negative_taxonomy.pdf / .png)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65), dpi=300)

# (a) Random Negative
axes[0].imshow(img_rand)
axes[0].set_title(r"(a) Random Negative ($r{=}40\%$)" + "\nLow Entropy: Routine Scene",
                  fontweight='bold', pad=5, fontsize=8.2)
axes[0].axis('off')
axes[0].text(320, 610, "Clean Silence (0 FP)", color='white', fontsize=7.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#2A9D8F', edgecolor='white', linewidth=0.8, alpha=0.95))

# (b) Hard Negative under Random Sampling
axes[1].imshow(img_hard)
axes[1].set_title(r"(b) Hard Negative / Random ($40\%$)" + "\nHigh Entropy: Context Binding",
                  fontweight='bold', pad=5, fontsize=8.2)
axes[1].axis('off')

# Hallucination bounding box: [0.0, 269.6, 92.1, 448.8] -> phone_use 0.68
r_fp = patches.Rectangle((0, 269.6), 92.1, 448.8 - 269.6, linewidth=2.2, edgecolor='#E63946', facecolor='none')
axes[1].add_patch(r_fp)
axes[1].text(5, 262, 'phone_use 0.68', color='white', fontsize=7.2, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.15', facecolor='#E63946', edgecolor='none', alpha=0.95))

axes[1].text(320, 610, "False Alarm Leakage", color='white', fontsize=7.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#E63946', edgecolor='white', linewidth=0.8, alpha=0.95))

# (c) Hard Negative under Active Curation
axes[2].imshow(img_hard)
axes[2].set_title(r"(c) Hard Negative Curated ($r{=}40\%$)" + "\nActive Supervision: Pruned Context",
                  fontweight='bold', pad=5, fontsize=8.2)
axes[2].axis('off')
axes[2].text(320, 610, "Hypothesis Suppressed (0 FP)", color='white', fontsize=7.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#2A9D8F', edgecolor='white', linewidth=0.8, alpha=0.95))

plt.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.85, bottom=0.03)
fig.savefig(os.path.join(out_dir, "fig_negative_taxonomy.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, "fig_negative_taxonomy.pdf"), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 2. Single-Column 2-Panel Version (fig_negative_taxonomy_col.pdf / .png)
# =============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(3.5, 2.05), dpi=300)

# (a) Random Negative
axes2[0].imshow(img_rand)
axes2[0].set_title(r"(a) Random ($r{=}40\%$)" + "\nLow-Entropy Scene",
                   fontweight='bold', pad=4, fontsize=7.2)
axes2[0].axis('off')
axes2[0].text(320, 610, "Clean Silence (0 FP)", color='white', fontsize=5.8, fontweight='bold', ha='center',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#2A9D8F', edgecolor='white', linewidth=0.6, alpha=0.95))

# (b) Hard-Mined Negative
axes2[1].imshow(img_hard)
axes2[1].set_title(r"(b) Hard-Mined ($40\%$)" + "\nHigh-Entropy Shadows",
                   fontweight='bold', pad=4, fontsize=7.2)
axes2[1].axis('off')
r_fp2 = patches.Rectangle((0, 269.6), 92.1, 448.8 - 269.6, linewidth=1.6, edgecolor='#E63946', facecolor='none')
axes2[1].add_patch(r_fp2)
axes2[1].text(5, 262, 'phone_use 0.68', color='white', fontsize=5.6, fontweight='bold',
              bbox=dict(boxstyle='square,pad=0.12', facecolor='#E63946', edgecolor='none', alpha=0.95))
axes2[1].text(320, 610, "False Alarm Leakage", color='white', fontsize=5.8, fontweight='bold', ha='center',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#E63946', edgecolor='white', linewidth=0.6, alpha=0.95))

plt.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.84, bottom=0.03)
fig2.savefig(os.path.join(out_dir, "fig_negative_taxonomy_col.png"), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(out_dir, "fig_negative_taxonomy_col.pdf"), bbox_inches='tight')
plt.close(fig2)

print("Generated both 3-panel (double-column) and 2-panel (single-column) publication figures.")
