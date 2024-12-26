import cv2
import torch
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set confidence threshold (optional)
model.conf = 0.5  # Default is 0.25

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# FPS calculation
prev_time = time.time()

# Initialize frame counter
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Process every 5th frame
    if frame_count % 5 == 0:
        # Perform detection
        results = model(frame)

        # Render results on the frame
        rendered_frame = results.render()[0]
    else:
        # If skipping frames, just display the unprocessed frame
        rendered_frame = frame

    # Increment frame counter
    frame_count += 1

    # Make the rendered frame writable
    rendered_frame = rendered_frame.copy()  # Fix for OpenCV putText error

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on the frame
    cv2.putText(rendered_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-Time Detection', rendered_frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
