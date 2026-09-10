import cv2

frames = [
    ("subj08_0184", "data/processed/RGB/images/subject_08/video_01/subject_08_video_01_frame_0184.jpg"),
    ("subj12_0249", "data/processed/RGB/images/subject_12/video_02/subject_12_video_02_frame_0249.jpg"),
    ("subj12_0207", "data/processed/RGB/images/subject_12/video_01/subject_12_video_01_frame_0207.jpg"),
    ("subj06_0055", "data/processed/RGB/images/subject_06/video_03/subject_06_video_03_frame_0055.jpg"),
    ("subj10_0453", "data/processed/RGB/images/subject_10/video_02/subject_10_video_02_frame_0453.jpg"),
    ("subj13_0047", "data/processed/RGB/images/subject_13/video_03/subject_13_video_03_frame_0047.jpg"),
    ("subj12_0032", "data/processed/RGB/images/subject_12/video_02/subject_12_video_02_frame_0032.jpg"),
]

for name, path in frames:
    img = cv2.imread(path)
    cv2.imwrite(f"scratch/raw_{name}.jpg", img)
print("Saved raw images")
