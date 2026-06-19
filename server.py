"""
FastAPI Server for FashionMNIST CNN Model Inference
====================================================
Provides endpoints to:
  - Upload a trained PyTorch CNN model (.pt file)
  - Predict FashionMNIST classes from uploaded images
  - Health check and class listing

Usage:
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from cnn import CNN  # user's CNN model class

# ---------------------------------------------------------------------------
# FashionMNIST class labels
# ---------------------------------------------------------------------------
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# ---------------------------------------------------------------------------
# Global model holder and image preprocessing
# ---------------------------------------------------------------------------
_model: CNN | None = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match the transform used in data_loader.py during training
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),                       # ensure single channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_model() -> CNN:
    """Return the loaded model, or raise if not loaded yet."""
    if _model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Upload a model first via POST /upload-model")
    return _model


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FashionMNIST CNN Predictor",
    description="Upload a PyTorch CNN model and classify FashionMNIST images.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok", "device": str(_device)}


@app.get("/classes")
async def list_classes() -> dict[str, list[str]]:
    """Return the list of class labels."""
    return {"classes": FASHION_MNIST_CLASSES}


@app.post("/predict-json")
async def predict_json(data: dict = Body(...)) -> dict[str, object]:
    """
    Accept JSON pixel data and return prediction.

    Body format:
      {"image": [v1, v2, ..., v784]}          // flat 784 array, values 0-255
      {"image": [[v11..v1n], ..., [vm1..vmn]]} // 2D 28x28 array, values 0-255

    Also supports batch:
      {"images": [[784], [784], ...]}           // multiple flat arrays
    """
    model = get_model()

    def parse_image(array: list) -> torch.Tensor:
        """Convert a flat 784 list or 2D 28x28 list into a preprocessed tensor."""
        if isinstance(array[0], list):
            # 2D array: [[row1], [row2], ...]
            h, w = len(array), len(array[0])
            img_np = __import__("numpy").array(array, dtype="uint8")
        else:
            flat_len = len(array)
            import math
            side = int(math.sqrt(flat_len))
            if side * side != flat_len:
                raise HTTPException(status_code=400, detail=f"Cannot infer image dimensions from flat array of length {flat_len}")
            img_np = __import__("numpy").array(array, dtype="uint8").reshape(side, side)
            h, w = side, side

        if h != 28 or w != 28:
            raise HTTPException(status_code=400, detail=f"Expected 28x28 image, got {h}x{w}")

        image = Image.fromarray(img_np, mode="L")
        return IMAGE_TRANSFORM(image)  # (1, 28, 28)

    # Single image
    if "image" in data:
        input_tensor = parse_image(data["image"]).unsqueeze(0).to(_device)  # (1, 1, 28, 28)

        with torch.inference_mode():
            logits: torch.Tensor = model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze().cpu().tolist()
            predicted_idx: int = int(logits.argmax(dim=1).item())

        return {
            "predicted_class": FASHION_MNIST_CLASSES[predicted_idx],
            "predicted_index": predicted_idx,
            "probabilities": {cls_name: round(prob, 4) for cls_name, prob in zip(FASHION_MNIST_CLASSES, probabilities)},
        }

    # Batch
    if "images" in data:
        images_list = data["images"]
        if not images_list:
            raise HTTPException(status_code=400, detail="Empty images list")

        tensors = [parse_image(img) for img in images_list]
        batch = torch.stack(tensors).to(_device)  # (N, 1, 28, 28)

        with torch.inference_mode():
            logits = model(batch)
            predicted_indices = logits.argmax(dim=1).tolist()

        return {
            "predictions": [
                {"index": i, "predicted_class": FASHION_MNIST_CLASSES[idx], "predicted_index": idx}
                for i, idx in enumerate(predicted_indices)
            ]
        }

    raise HTTPException(status_code=400, detail="Body must contain 'image' or 'images' key")

