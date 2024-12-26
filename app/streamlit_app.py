import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# App title
st.title("Real-Time Object Detection System")

# File uploader
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