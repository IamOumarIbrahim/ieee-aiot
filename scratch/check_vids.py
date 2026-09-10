import os
import glob

# Let's inspect label files in video_01 across all subjects to see what actions occur
for sub in ["subject_01", "subject_02", "subject_03", "subject_04", "subject_05"]:
    for v in ["video_01", "video_02", "video_03", "video_04", "video_05"]:
        lbl_files = glob.glob(f"data/processed/RGB/labels/{sub}/{v}/*.txt")
        non_empty = [f for f in lbl_files if os.path.getsize(f) > 0]
        print(f"{sub} {v}: total={len(lbl_files)}, non_empty={len(non_empty)}")
