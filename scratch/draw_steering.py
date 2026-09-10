import cv2
import json

with open("scratch/candidates.json") as f:
    data = json.load(f)

for idx in [0, 1, 4, 7, 8, 10, 17, 20]:
    d = data[idx]
    img = cv2.imread(d["img"])
    
    # Draw YOLO11 0% boxes (red)
    img11 = img.copy()
    for label, conf, box in d["yolo11_0"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img11, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img11, f"{label} {conf:.2f}", (x1, max(y1-8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw YOLO26 0% boxes (blue)
    img26 = img.copy()
    for label, conf, box in d["yolo26_0"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img26, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(img26, f"{label} {conf:.2f}", (x1, max(y1-8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    cv2.imwrite(f"scratch/vis_{idx}_11.jpg", img11)
    cv2.imwrite(f"scratch/vis_{idx}_26.jpg", img26)
    cv2.imwrite(f"scratch/vis_{idx}_orig.jpg", img)
    print(f"Candidate {idx}: {d['img']}")
