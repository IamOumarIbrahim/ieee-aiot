import cv2

img = cv2.imread("data/processed/RGB/images/subject_01/video_02/subject_01_video_02_frame_0210.jpg")
cv2.imwrite("scratch/subj01_0210_raw.jpg", img)

# draw box [0, 256, 110, 449]
img_b = img.copy()
cv2.rectangle(img_b, (0, 256), (110, 449), (0, 0, 255), 3)
cv2.putText(img_b, "phone_use 0.60", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite("scratch/subj01_0210_box.jpg", img_b)
