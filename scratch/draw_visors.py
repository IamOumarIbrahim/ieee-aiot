import cv2

frames = [
    ("subj05_0018", "data/processed/RGB/images/subject_05/video_03/subject_05_video_03_frame_0018.jpg", [("hand_over_mouth", 0.29, [232, 1, 640, 187])]),
    ("subj13_0047", "data/processed/RGB/images/subject_13/video_03/subject_13_video_03_frame_0047.jpg", [("drinking", 0.67, [40, 134, 438, 432])]),
    ("subj07_0040", "data/processed/RGB/images/subject_07/video_03/subject_07_video_03_frame_0040.jpg", [("yawning", 0.83, [274, 443, 355, 484])]),
    ("subj07_0036", "data/processed/RGB/images/subject_07/video_03/subject_07_video_03_frame_0036.jpg", [("yawning", 0.84, [230, 444, 306, 495])]),
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
