import cv2
import json

with open("scratch/candidates.json") as f:
    data = json.load(f)

# Let us draw boxes for candidates 41, 57, 70, 91, 119, 60, 46
candidates_to_draw = [41, 57, 70, 91, 119, 46, 69]

for idx in candidates_to_draw:
    d = data[idx]
    img = cv2.imread(d["img"])
    h, w, _ = img.shape
    
    # Draw YOLO11 0% boxes
    img11 = img.copy()
    for label, conf, box in d["yolo11_0"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img11, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img11, f"{label} {conf:.2f}", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw YOLO26 0% boxes
    img26 = img.copy()
    for label, conf, box in d["yolo26_0"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img26, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img26, f"{label} {conf:.2f}", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    out_path = f"scratch/vis_cand_{idx}.jpg"
    # concatenate horizontally: original, yolo11_0, yolo26_0
    combo = cv2.hconcat([img, img11, img26])
    cv2.imwrite(out_path, combo)
    print(f"Saved {out_path} for {d['img']}")

