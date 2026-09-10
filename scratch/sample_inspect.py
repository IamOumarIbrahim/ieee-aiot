import cv2
import numpy as np
from pathlib import Path
import random

# Sample 20 frames across subjects and videos
sample_frames = [
    "data/processed/RGB/images/subject_01/video_01/subject_01_video_01_frame_0004.jpg",
    "data/processed/RGB/images/subject_02/video_03/subject_02_video_03_frame_0010.jpg",
    "data/processed/RGB/images/subject_03/video_02/subject_03_video_02_frame_0050.jpg",
    "data/processed/RGB/images/subject_04/video_04/subject_04_video_04_frame_0080.jpg",
    "data/processed/RGB/images/subject_05/video_05/subject_05_video_05_frame_0005.jpg",
    "data/processed/RGB/images/subject_06/video_01/subject_06_video_01_frame_0020.jpg",
    "data/processed/RGB/images/subject_07/video_02/subject_07_video_02_frame_0040.jpg",
    "data/processed/RGB/images/subject_08/video_03/subject_08_video_03_frame_0060.jpg",
    "data/processed/RGB/images/subject_09/video_04/subject_09_video_04_frame_0090.jpg",
    "data/processed/RGB/images/subject_10/video_01/subject_10_video_01_frame_0030.jpg",
    "data/processed/RGB/images/subject_11/video_02/subject_11_video_02_frame_0070.jpg",
    "data/processed/RGB/images/subject_12/video_03/subject_12_video_03_frame_0015.jpg",
    "data/processed/RGB/images/subject_13/video_04/subject_13_video_04_frame_0055.jpg",
    "data/processed/RGB/images/subject_14/video_05/subject_14_video_05_frame_0025.jpg",
]

for p in sample_frames:
    img = cv2.imread(p)
    if img is not None:
        mean_val = img.mean()
        std_val = img.std()
        print(f"{p}: shape={img.shape}, mean={mean_val:.1f}, std={std_val:.1f}")
    else:
        print(f"Missing: {p}")
