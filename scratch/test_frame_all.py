from ultralytics import YOLO
import json

img_path = "data/processed/RGB/images/subject_08/video_02/subject_08_video_02_frame_0207.jpg"
names = {0: "yawning", 1: "hand_over_mouth", 2: "drinking", 3: "phone_use"}

for model_name in ["yolo11n", "yolo26n", "yolo12n", "yolov10n"]:
    m0 = YOLO(f"runs/{model_name}_ratio_sweep/train_00_pos_only/weights/best.pt")
    m80 = YOLO(f"runs/{model_name}_ratio_sweep/train_80_max_neg/weights/best.pt")
    
    r0 = m0(img_path, conf=0.25, verbose=False)[0]
    r80 = m80(img_path, conf=0.25, verbose=False)[0]
    
    dets0 = [(names[int(b.cls[0])], float(b.conf[0]), [round(x, 1) for x in b.xyxy[0].tolist()]) for b in r0.boxes]
    dets80 = [(names[int(b.cls[0])], float(b.conf[0]), [round(x, 1) for x in b.xyxy[0].tolist()]) for b in r80.boxes]
    
    print(f"=== {model_name} ===")
    print("  0% :", dets0)
    print("  80%:", dets80)
