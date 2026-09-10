import cv2

img = cv2.imread("data/processed/RGB/images/subject_03/video_01/subject_03_video_01_frame_0116.jpg")
cv2.imwrite("scratch/subj03_0116_raw.jpg", img)

# draw box [132.2, 375.0, 555.3, 640.0]
img_box = img.copy()
cv2.rectangle(img_box, (132, 375), (555, 640), (0, 0, 255), 3)
cv2.putText(img_box, "drinking 0.48", (135, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imwrite("scratch/subj03_0116_box.jpg", img_box)
print("Saved subj03_0116")
