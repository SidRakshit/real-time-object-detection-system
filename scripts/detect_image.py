import torch
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set the directory to save all outputs
output_dir = Path('data/output/')
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# Directory containing input images
img_dir = Path('data/images/')

# Perform inference on all images in the folder
for img_path in img_dir.glob('*.jpg'):
    results = model(str(img_path))  # Perform detection
    for img_result in results.imgs:  # Loop through processed images
        save_path = output_dir / img_path.name
        results.render()  # Render the bounding boxes on the image
        results.imgs[0].save(save_path)  # Save processed image to a single folder
        print(f"Processed: {img_path.name}")
