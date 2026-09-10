from ultralytics import YOLO
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np

img_path = 'data/processed/RGB/images/subject_08/video_02/subject_08_video_02_frame_0207.jpg'
orig_img = Image.open(img_path).convert('RGB')
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

m26 = YOLO('runs/yolo26n_ratio_sweep/train_00_pos_only/weights/best.pt').model.cuda()
m12 = YOLO('runs/yolo12n_ratio_sweep/train_00_pos_only/weights/best.pt').model.cuda()

# Gradient for YOLO26n
t_im26 = t_im.clone().detach().requires_grad_(True)
_, cls26, saved26 = forward_model(m26, t_im26)
loss26 = sum([F.binary_cross_entropy_with_logits(c, torch.zeros_like(c), reduction='sum') for c in cls26])
loss26.backward()
g26 = t_im26.grad.detach().abs().squeeze(0).cpu().numpy().max(axis=0)

# Gradient for YOLO12n
t_im12 = t_im.clone().detach().requires_grad_(True)
_, cls12, saved12 = forward_model(m12, t_im12)
loss12 = sum([F.binary_cross_entropy_with_logits(c, torch.zeros_like(c), reduction='sum') for c in cls12])
loss12.backward()
g12 = t_im12.grad.detach().abs().squeeze(0).cpu().numpy().max(axis=0)

# Check P5 activations
p26_act = torch.sigmoid(cls26[2]).squeeze(0).max(dim=0)[0].detach().cpu().numpy()
p12_act = torch.sigmoid(cls12[2]).squeeze(0).max(dim=0)[0].detach().cpu().numpy()

# Check saved intermediate feature maps
feat26 = saved26[19].squeeze(0).abs().mean(dim=0).detach().cpu().numpy()
feat12 = saved12[17].squeeze(0).abs().mean(dim=0).detach().cpu().numpy()

print(f"g26 shape: {g26.shape}, max: {g26.max():.4f}")
print(f"g12 shape: {g12.shape}, max: {g12.max():.4f}")
print(f"p26_act shape: {p26_act.shape}, max: {p26_act.max():.4f}")
print(f"p12_act shape: {p12_act.shape}, max: {p12_act.max():.4f}")
print(f"feat26 shape: {feat26.shape}, max: {feat26.max():.4f}")
print(f"feat12 shape: {feat12.shape}, max: {feat12.max():.4f}")
