"""
YOLOv8 Training Script
References: https://youtu.be/WgPbbWmnXJ8
This script demonstrates how to train a YOLOv8 model using the Ultralytics library.
"""

from ultralytics import YOLO
import torch

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

if __name__ == "__main__":

    # Select device for training
    device = get_device()

    # Use YOLO's default
    PRE_TRAINED_MODEL = 'models/yolov8n.pt'

    # Load a model
    model = YOLO(PRE_TRAINED_MODEL)

    # Train the model
    results = model.train(data='config.yaml', epochs=100, imgsz=640, device=device)
