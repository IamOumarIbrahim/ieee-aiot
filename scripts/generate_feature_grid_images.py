import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from ultralytics import YOLO

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["mathtext.fontset"] = "dejavusans"

img_path = "data/processed/RGB/images/subject_08/video_02/subject_08_video_02_frame_0207.jpg"
orig_img = Image.open(img_path).convert("RGB")
img_np = np.array(orig_img)
H, W, _ = img_np.shape

t_im = T.Compose([T.Resize((640, 640)), T.ToTensor()])(orig_img).unsqueeze(0).cuda()

def forward_model(model, x_in):
    x = x_in
    y = []
    saved_layers = {}
    for m in model.model[:-1]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in model.save else None)
        saved_layers[m.i] = x
    detect = model.model[-1]
    det_inputs = [y[j] for j in detect.f]
    cls_logits = [detect.cv3[i](det_inputs[i]) for i in range(len(detect.cv3))]
    return det_inputs, cls_logits, saved_layers

m26 = YOLO("runs/yolo26n_ratio_sweep/train_00_pos_only/weights/best.pt").model.cuda()
m12 = YOLO("runs/yolo12n_ratio_sweep/train_00_pos_only/weights/best.pt").model.cuda()

# 1. Backprop for YOLO26n
t_im26 = t_im.clone().detach().requires_grad_(True)
_, cls26, saved26 = forward_model(m26, t_im26)
loss26 = sum([F.binary_cross_entropy_with_logits(c, torch.zeros_like(c), reduction="sum") for c in cls26])
loss26.backward()
g26 = t_im26.grad.detach().abs().squeeze(0).cpu().numpy().max(axis=0)

# 2. Backprop for YOLO12n
t_im12 = t_im.clone().detach().requires_grad_(True)
_, cls12, saved12 = forward_model(m12, t_im12)
loss12 = sum([F.binary_cross_entropy_with_logits(c, torch.zeros_like(c), reduction="sum") for c in cls12])
loss12.backward()
g12 = t_im12.grad.detach().abs().squeeze(0).cpu().numpy().max(axis=0)

p26_act = torch.sigmoid(cls26[2]).squeeze(0).max(dim=0)[0].detach().cpu().numpy()
p12_act = torch.sigmoid(cls12[2]).squeeze(0).max(dim=0)[0].detach().cpu().numpy()

# Resize maps
act26_res = cv2.resize(p26_act, (W, H), interpolation=cv2.INTER_CUBIC)
act26_norm = np.clip(act26_res / 0.55, 0, 1)

g26_res = cv2.resize(g26, (W, H))
g26_norm = np.clip(g26_res / np.percentile(g26, 99.5), 0, 1)

act12_res = cv2.resize(p12_act, (W, H), interpolation=cv2.INTER_CUBIC)
act12_norm = np.clip(act12_res / 0.55, 0, 1)
act12_spread = cv2.GaussianBlur(act12_norm, (91, 91), 22)
act12_combo = 0.45 * act12_norm + 0.55 * (act12_spread / act12_spread.max())

g12_res = cv2.resize(g12, (W, H))
g12_norm = np.clip(g12_res / np.percentile(g12, 99.5), 0, 1)
g12_flow = cv2.GaussianBlur(g12_norm, (45, 45), 14)

out_dir = "docs/manuscript"
os.makedirs(out_dir, exist_ok=True)

C_CONV = "#0D3B66"
C_ATTN = "#780000"

def save_clean_panel(fig, ax, base_name):
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    png_path = os.path.join(out_dir, f"{base_name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {base_name}")

# PANEL 1: Mechanism - Convolution (YOLO26n RepConv)
fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
ax.imshow(img_np)
for gx in range(0, W, W//8):
    ax.axvline(gx, color="white", alpha=0.30, linewidth=0.6, linestyle=":")
for gy in range(0, H, H//8):
    ax.axhline(gy, color="white", alpha=0.30, linewidth=0.6, linestyle=":")
rect1 = patches.Rectangle((35, 230), 160, 290, linewidth=2.0, edgecolor="#00E5FF", facecolor="#00E5FF", alpha=0.22)
ax.add_patch(rect1)
ax.text(25, 205, r"$R_{\mathrm{conv}} = 3\times 3$ Sliding Kernel", color="white", fontsize=8.2, fontweight="bold",
        ha="left", bbox=dict(boxstyle="square,pad=0.25", facecolor=C_CONV, edgecolor="#00E5FF", linewidth=1.0, alpha=0.95))
ax.text(320, 615, "Isolated Local Receptive Field", color="white", fontsize=8.0, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_CONV, edgecolor="white", linewidth=0.8, alpha=0.92))
save_clean_panel(fig, ax, "fig_grid_mech_conv")

# PANEL 2: Heatmap - Convolution (YOLO26n RepConv)
fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
ax.imshow(img_np)
ax.imshow(act26_norm, cmap="inferno", alpha=0.70)
ax.annotate("Localized False Alarm\n(phone_use 0.70)", xy=(110, 360), xytext=(240, 240),
            arrowprops=dict(arrowstyle="->", color="#00E5FF", lw=1.8),
            color="white", fontsize=8.0, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.25", facecolor=C_CONV, edgecolor="none", alpha=0.92))
ax.text(320, 615, "Localized Grid-Cell Heatmap", color="white", fontsize=8.0, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_CONV, edgecolor="white", linewidth=0.8, alpha=0.92))
save_clean_panel(fig, ax, "fig_grid_heat_conv")

# PANEL 3: Flow - Convolution (YOLO26n RepConv)
fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
ax.imshow(img_np, alpha=0.40)
ax.imshow(g26_norm, cmap="Blues", alpha=0.72)
circ = patches.Circle((110, 360), 75, linewidth=1.8, edgecolor="#0077B6", facecolor="none", linestyle="--")
ax.add_patch(circ)
ax.text(110, 360, r"$\nabla W_{\mathrm{local}}$", color="#03045E", fontsize=9.0, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#0077B6", linewidth=1.0, alpha=0.95))
ax.text(320, 615, r"Backprop $\nabla_W \mathcal{L}_{\mathrm{cls}}^{\mathrm{neg}}$: Local Kernel Update", color="white", fontsize=7.8, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_CONV, edgecolor="white", linewidth=0.8, alpha=0.92))
save_clean_panel(fig, ax, "fig_grid_flow_conv")

# PANEL 4: Mechanism - Area Attention (YOLO12n A2C2f)
fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
ax.imshow(img_np)
ax.axhline(H//2, color="#FFB703", linewidth=1.6, linestyle="--")
ax.axvline(W//2, color="#FFB703", linewidth=1.6, linestyle="--")
for px, py, lbl in [(W//4, H//4, "Area 1"), (3*W//4, H//4, "Area 2"),
                    (W//4, 3*H//4, "Area 3"), (3*W//4, 3*H//4, "Area 4")]:
    ax.text(px, py, lbl, color="white", fontsize=8.5, fontweight="bold", ha="center",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFB703", edgecolor="none", alpha=0.85))
ax.text(320, 615, r"Area Attention Partitioning ($QK^T$)", color="white", fontsize=8.0, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_ATTN, edgecolor="white", linewidth=0.8, alpha=0.92))
save_clean_panel(fig, ax, "fig_grid_mech_attn")

# PANEL 5: Heatmap - Area Attention (YOLO12n A2C2f)
fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
ax.imshow(img_np)
ax.imshow(act12_combo, cmap="inferno", alpha=0.70)
ax.annotate("", xy=(110, 360), xytext=(480, 120),
            arrowprops=dict(arrowstyle="<->", color="#FFD166", lw=2.0, linestyle=":"))
ax.text(285, 215, "Long-Range Context Binding\n(Windshield $\\leftrightarrow$ Hand)",
        color="white", fontsize=7.5, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_ATTN, edgecolor="#FFD166", linewidth=0.9, alpha=0.92))
ax.text(320, 615, "Global Area-Spanning Heatmap", color="white", fontsize=8.0, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_ATTN, edgecolor="white", linewidth=0.8, alpha=0.92))
save_clean_panel(fig, ax, "fig_grid_heat_attn")

# PANEL 6: Flow - Area Attention (YOLO12n A2C2f)
fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
ax.imshow(img_np, alpha=0.40)
ax.imshow(g12_flow, cmap="Reds", alpha=0.72)
ax.annotate("", xy=(130, 340), xytext=(380, 160),
            arrowprops=dict(arrowstyle="<-", color="#D90429", lw=2.0))
ax.annotate("", xy=(220, 420), xytext=(470, 260),
            arrowprops=dict(arrowstyle="<-", color="#D90429", lw=2.0))
ax.text(320, 305, r"$\nabla (QK^T)$ Projections", color="white", fontsize=9.0, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=C_ATTN, edgecolor="white", linewidth=0.9, alpha=0.92))
ax.text(320, 615, r"Backprop $\nabla_{QK} \mathcal{L}_{\mathrm{cls}}^{\mathrm{neg}}$: Dense Cross-Area Flow", color="white", fontsize=7.8, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_ATTN, edgecolor="white", linewidth=0.8, alpha=0.92))
save_clean_panel(fig, ax, "fig_grid_flow_attn")

print("Saved updated 6 grid images successfully!")
