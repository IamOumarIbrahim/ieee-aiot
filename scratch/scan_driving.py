import os
import json
from ultralytics import YOLO

with open("data/processed/RGB/yolo/test.txt") as f:
    lines = [l.strip() for l in f if l.strip()]

# Filter for video_01 and video_03 frames in test set
driving_frames = []
for l in lines:
    if "video_01" in l or "video_03" in l:
        img_p = os.path.normpath(os.path.join("data/processed/RGB/yolo", l))
        lbl_p = img_p.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if not (os.path.exists(lbl_p) and os.path.getsize(lbl_p) > 0):
            driving_frames.append(img_p)

print(f"Total pure driving background frames in test set: {len(driving_frames)}")

m11_0 = YOLO("runs/yolo11n_ratio_sweep/train_00_pos_only/weights/best.pt")
m11_80 = YOLO("runs/yolo11n_ratio_sweep/train_80_max_neg/weights/best.pt")
m26_0 = YOLO("runs/yolo26n_ratio_sweep/train_00_pos_only/weights/best.pt")
m26_80 = YOLO("runs/yolo26n_ratio_sweep/train_80_max_neg/weights/best.pt")

names = {0: "yawning", 1: "hand_over_mouth", 2: "drinking", 3: "phone_use"}

results = []
for img in driving_frames:
    r11_0 = m11_0(img, conf=0.25, verbose=False)[0]
    r11_80 = m11_80(img, conf=0.25, verbose=False)[0]
    r26_0 = m26_0(img, conf=0.25, verbose=False)[0]
    r26_80 = m26_80(img, conf=0.25, verbose=False)[0]
    
    has_0 = len(r11_0.boxes) > 0 or len(r26_0.boxes) > 0
    is_silent_80 = len(r11_80.boxes) == 0 and len(r26_80.boxes) == 0
    if has_0 and is_silent_80:
        b11 = [(names[int(b.cls[0])], float(b.conf[0]), [round(x, 1) for x in b.xyxy[0].tolist()]) for b in r11_0.boxes]
        b26 = [(names[int(b.cls[0])], float(b.conf[0]), [round(x, 1) for x in b.xyxy[0].tolist()]) for b in r26_0.boxes]
        results.append({
            "img": img,
            "yolo11_0": b11,
            "yolo26_0": b26
        })

print(f"Found {len(results)} driving frames with 0% FPs and silent 80%!")
with open("scratch/driving_fps.json", "w") as f:
    json.dump(results, f, indent=2)

for r in results[:15]:
    print(r["img"])
    if r["yolo11_0"]: print("  11:", r["yolo11_0"])
    if r["yolo26_0"]: print("  26:", r["yolo26_0"])
