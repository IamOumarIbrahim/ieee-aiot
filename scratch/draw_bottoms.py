import cv2

frames = [
    ("subj11_0311", "data/processed/RGB/images/subject_11/video_02/subject_11_video_02_frame_0311.jpg", [("phone_use", 0.64, [0, 367, 246, 640])]),
    ("subj09_0175", "data/processed/RGB/images/subject_09/video_02/subject_09_video_02_frame_0175.jpg", [("drinking", 0.34, [179, 611, 423, 640])]),
    ("subj03_0122", "data/processed/RGB/images/subject_03/video_01/subject_03_video_01_frame_0122.jpg", [("drinking", 0.42, [0, 605, 252, 640])]),
]

for name, path, boxes in frames:
    img = cv2.imread(path)
    cv2.imwrite(f"scratch/{name}_raw.jpg", img)
    img_b = img.copy()
    for lbl, conf, box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_b, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_b, f"{lbl} {conf:.2f}", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(f"scratch/{name}_box.jpg", img_b)

print("Saved frames")
