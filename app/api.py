from fastapi import FastAPI, File, UploadFile
import torch
import cv2
from PIL import Image
import tempfile
from pathlib import Path
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLOv5 Object Detection API!"}

# Endpoint for object detection on images
@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded file
    image = Image.open(file.file)

    # Perform detection
    results = model(image)

    # Convert results to JSON
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    return {"detections": detections}

# Endpoint for object detection on videos
@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    # Path for output video
    output_path = str(Path(temp_video_path).stem) + "_processed.mp4"

    # Open the video file
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Render results
        rendered_frame = results.render()[0]
        out.write(rendered_frame)

    cap.release()
    out.release()

    # Return processed video file
    return {"message": "Video processed successfully!", "output_path": output_path}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
