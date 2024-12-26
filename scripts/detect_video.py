import cv2
import torch
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Path to input video
video_path = Path('data/videos/dan-dan-dan-sample-video.mp4')
output_path = Path('data/output/processed_video.avi')

# Open video file
cap = cv2.VideoCapture(str(video_path))

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create a VideoWriter to save the output
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Render results on the frame
    rendered_frame = results.render()[0]

    # Write the frame to the output file
    out.write(rendered_frame)

cap.release()
out.release()
print(f"Processed video saved to {output_path}")
