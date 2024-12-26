import streamlit as st
import torch
import cv2
from PIL import Image
import tempfile
import numpy as np
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# App title
st.title("Real-Time Object Detection System")

# Choose between image and video processing
option = st.radio("Select file type:", ("Image", "Video"))

if option == "Image":
    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load the image
        image = Image.open(uploaded_file)

        # Perform detection
        results = model(image)

        # Display the original image
        st.image(np.array(image), caption="Uploaded Image", use_column_width=True)

        # Display the results
        st.image(np.array(results.render()[0]), caption="Detection Results", use_column_width=True)

        # Show detection details
        st.write("Detections:")
        st.write(results.pandas().xyxy[0])

elif option == "Video":
    # File uploader for video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

        # Path for output video
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = output_file.name

        # Open the video file
        cap = cv2.VideoCapture(temp_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process video frame by frame
        st.text("Processing video... This may take a while.")
        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            results = model(frame)

            # Render results
            rendered_frame = results.render()[0]
            out.write(rendered_frame)

            # Update progress bar
            frame_processed += 1
            progress_bar.progress(min(frame_processed / frame_count, 1.0))

        cap.release()
        out.release()

        # Display the processed video
        st.video(output_path)
        st.success("Video processing complete!")
