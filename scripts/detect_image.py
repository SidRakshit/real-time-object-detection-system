import torch
from pathlib import Path
from PIL import Image  # Import the Image module from Pillow


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
    # print(dir(results))
    for img_result in results.ims:  # Loop through processed images
        save_path = output_dir / img_path.name
        results.render()  # Render the bounding boxes on the image
        image = Image.fromarray(results.ims[0])
        image.save(save_path)
        print(f"Processed: {img_path.name}")

