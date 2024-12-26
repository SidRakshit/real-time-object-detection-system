from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import uvicorn
from pathlib import Path

# Initialize the FastAPI app
app = FastAPI()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Endpoint for object detection on an image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    image = Image.open(file.file)

    # Perform detection
    results = model(image)

    # Convert results to JSON
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    return {"detections": detections}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)