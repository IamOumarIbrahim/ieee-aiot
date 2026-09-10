import torch
from ultralytics import YOLO

# Load baseline model (0% negatives) and curated model
m_base = YOLO("runs/yolo11n_ratio_sweep/train_00_pos_only/weights/best.pt")
m_rand = YOLO("runs/yolo11n_ratio_sweep/train_40_mod_neg/weights/best.pt")
m_hard = YOLO("runs/yolo11n_ratio_sweep/train_curated_yolo11n_best_curated/weights/best.pt")

img_random = "data/processed/RGB/images/subject_01/video_01/subject_01_video_01_frame_0004.jpg"
img_hard = "data/processed/RGB/images/subject_01/video_02/subject_01_video_02_frame_0210.jpg"

print("--- Testing Random Negative (Flat, Featureless) ---")
res_base_r = m_base.predict(img_random, conf=0.25, verbose=False)[0]
print(f"Baseline on Random Neg: {len(res_base_r.boxes)} boxes")

print("\n--- Testing Hard-Mined Negative (Complex Shadows/Lighting) ---")
res_base_h = m_base.predict(img_hard, conf=0.25, verbose=False)[0]
print(f"Baseline on Hard Neg: {len(res_base_h.boxes)} boxes")
for b in res_base_h.boxes:
    cls_name = m_base.names[int(b.cls.item())]
    conf = float(b.conf.item())
    box = b.xyxy[0].tolist()
    print(f"   Hallucination: {cls_name} {conf:.2f} at {box}")

res_rand_h = m_rand.predict(img_hard, conf=0.25, verbose=False)[0]
print(f"Random (40%) on Hard Neg: {len(res_rand_h.boxes)} boxes")
for b in res_rand_h.boxes:
    cls_name = m_rand.names[int(b.cls.item())]
    conf = float(b.conf.item())
    box = b.xyxy[0].tolist()
    print(f"   Hallucination: {cls_name} {conf:.2f} at {box}")

res_hard_h = m_hard.predict(img_hard, conf=0.25, verbose=False)[0]
print(f"Hard-Curated (40%) on Hard Neg: {len(res_hard_h.boxes)} boxes")
for b in res_hard_h.boxes:
    cls_name = m_hard.names[int(b.cls.item())]
    conf = float(b.conf.item())
    box = b.xyxy[0].tolist()
    print(f"   Hallucination: {cls_name} {conf:.2f} at {box}")
