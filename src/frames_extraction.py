# Conceptual frame extraction snippet
import cv2
import os

video_path = "Raw_Data/Video5.mp4"
output_folder = "Extracted_Frames/Video5_Frames"
os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)
frame_interval = 1 # Extract 1 frame every 10 frames (adjust based on video FPS and activity)
frame_count = 0
saved_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        cv2.imwrite(os.path.join(output_folder, f"frame_{saved_count:05d}.jpg"), frame)
        saved_count += 1
    frame_count += 1
cap.release()
print(f"Extracted {saved_count} frames.")