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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from CNN import FashionCNN  # user's CNN model class

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
_model: FashionCNN | None = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match the transform used in data_loader.py during training
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),                       # ensure single channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_model() -> FashionCNN:
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


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Upload a PyTorch model file (.pt or .pth).

    Supports both:
      - Full model:     torch.save(model, 'model.pt')
      - State dict:     torch.save(model.state_dict(), 'model.pt')
    """
    global _model

    if not file.filename or not file.filename.endswith((".pt", ".pth")):
        raise HTTPException(status_code=400, detail="File must be a .pt or .pth PyTorch model file")

    contents = await file.read()
    buffer = io.BytesIO(contents)
    checkpoint = torch.load(buffer, map_location=_device, weights_only=False)

    # Instantiate model
    model = FashionCNN(num_classes=len(FASHION_MNIST_CLASSES))

    if isinstance(checkpoint, dict):
        # Try to detect state_dict keys — some savers wrap it
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif any(k.startswith("conv") or k.startswith("fc") or k.startswith("bn") for k in checkpoint.keys()):
            model.load_state_dict(checkpoint)
        else:
            raise HTTPException(
                status_code=400,
                detail="Uploaded dict does not appear to be a model state_dict. "
                       "Keys found: " + ", ".join(list(checkpoint.keys())[:10]),
            )
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        raise HTTPException(status_code=400, detail="Unrecognised file format. Expected nn.Module or state_dict.")

    model.to(_device)
    model.eval()
    _model = model

    return {"message": "Model loaded successfully", "device": str(_device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    """
    Upload a 28x28 grayscale FashionMNIST image and receive a prediction.

    Returns the predicted class name and per-class probabilities.
    """
    model = get_model()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read and preprocess image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    input_tensor: torch.Tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(_device)  # (1, 1, 28, 28)

    # Inference
    with torch.inference_mode():
        logits: torch.Tensor = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze().cpu().tolist()
        predicted_idx: int = int(logits.argmax(dim=1).item())

    return {
        "predicted_class": FASHION_MNIST_CLASSES[predicted_idx],
        "predicted_index": predicted_idx,
        "probabilities": {cls_name: round(prob, 4) for cls_name, prob in zip(FASHION_MNIST_CLASSES, probabilities)},
    }


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, object]:
    """Predict multiple images in one request (uses batched inference)."""
    model = get_model()

    tensors: list[torch.Tensor] = []
    filenames: list[str] = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        tensors.append(IMAGE_TRANSFORM(image))
        filenames.append(file.filename or "unknown")

    if not tensors:
        raise HTTPException(status_code=400, detail="No valid images provided")

    batch = torch.stack(tensors).to(_device)  # (N, 1, 28, 28)

    with torch.inference_mode():
        logits = model(batch)
        predicted_indices = logits.argmax(dim=1).tolist()

    return {
        "predictions": [
            {"filename": name, "predicted_class": FASHION_MNIST_CLASSES[idx], "predicted_index": idx}
            for name, idx in zip(filenames, predicted_indices)
        ]
    }


# ---------------------------------------------------------------------------
# Startup: auto-load model if MODEL_PATH env var is set
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:  # noqa: D401
    """Attempt to auto-load a model from $MODEL_PATH on startup (optional)."""
    global _model
    model_path = Path(__import__("os").environ.get("MODEL_PATH", ""))
    if model_path.is_file():
        checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
        model = FashionCNN(num_classes=len(FASHION_MNIST_CLASSES))
        if isinstance(checkpoint, dict):
            key = next((k for k in ("model_state_dict", "state_dict") if k in checkpoint), None)
            model.load_state_dict(checkpoint[key] if key else checkpoint)
        elif isinstance(checkpoint, nn.Module):
            model = checkpoint
        model.to(_device).eval()
        _model = model
        print(f"Model auto-loaded from {model_path}")
