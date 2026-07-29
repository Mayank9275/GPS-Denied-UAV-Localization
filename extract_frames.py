import cv2
import os

video_path = "Data/Vedio_1/flight_video.mp4"
output_folder = "Data/Vedio_1/Extracted-frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save only every 10th frame
    if frame_count % 10 == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Total video frames: {frame_count}")
print(f"Frames saved: {saved_count}")