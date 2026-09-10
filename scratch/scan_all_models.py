import os
import json
from ultralytics import YOLO

with open("data/processed/RGB/yolo/test.txt") as f:
    lines = [l.strip() for l in f if l.strip()]

neg_imgs = []
for l in lines:
    img_p = os.path.normpath(os.path.join("data/processed/RGB/yolo", l))
    lbl_p = img_p.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    if not (os.path.exists(lbl_p) and os.path.getsize(lbl_p) > 0):
        neg_imgs.append(img_p)

models_0 = {
    "yolo11n": YOLO("runs/yolo11n_ratio_sweep/train_00_pos_only/weights/best.pt"),
    "yolo26n": YOLO("runs/yolo26n_ratio_sweep/train_00_pos_only/weights/best.pt"),
    "yolo12n": YOLO("runs/yolo12n_ratio_sweep/train_00_pos_only/weights/best.pt"),
    "yolov10n": YOLO("runs/yolov10n_ratio_sweep/train_00_pos_only/weights/best.pt")
}

models_80 = {
    "yolo11n": YOLO("runs/yolo11n_ratio_sweep/train_80_max_neg/weights/best.pt"),
    "yolo26n": YOLO("runs/yolo26n_ratio_sweep/train_80_max_neg/weights/best.pt"),
    "yolo12n": YOLO("runs/yolo12n_ratio_sweep/train_80_max_neg/weights/best.pt"),
    "yolov10n": YOLO("runs/yolov10n_ratio_sweep/train_80_max_neg/weights/best.pt")
}

names = {0: "yawning", 1: "hand_over_mouth", 2: "drinking", 3: "phone_use"}

# Let us check which negative images produce false alarms on windshield/window, steering wheel, etc.
detections_by_img = {}

# Sample 300 negative frames
for img in neg_imgs[:400]:
    frame_dets = {"0": {}, "80": {}}
    has_any_0 = False
    has_any_80 = False
    for mname in ["yolo11n", "yolo26n", "yolo12n", "yolov10n"]:
        res0 = models_0[mname](img, conf=0.25, verbose=False)[0]
        res80 = models_80[mname](img, conf=0.25, verbose=False)[0]
        if len(res0.boxes) > 0:
            has_any_0 = True
            frame_dets["0"][mname] = [(names[int(b.cls[0])], float(b.conf[0]), [round(x, 1) for x in b.xyxy[0].tolist()]) for b in res0.boxes]
        if len(res80.boxes) > 0:
            has_any_80 = True
            frame_dets["80"][mname] = [(names[int(b.cls[0])], float(b.conf[0]), [round(x, 1) for x in b.xyxy[0].tolist()]) for b in res80.boxes]
    if has_any_0 and not has_any_80:
        detections_by_img[img] = frame_dets

print(f"Found {len(detections_by_img)} frames where 0% had FPs and 80% was silent across all 4 models!")
with open("scratch/all_models_silent80.json", "w") as f:
    json.dump(detections_by_img, f, indent=2)

