# FashionMNIST CNN Model Server

FastAPI server that serves a PyTorch CNN model for FashionMNIST image classification. Supports model upload, single-image prediction, and batch prediction — all containerised with Docker.

## Quick Start

### 1. Build the Docker image

```bash
docker build -t fashion-cnn-server .
```

### 2. Run the container

```bash
docker run -p 8000:8000 fashion-cnn-server
```

The server is now live at **http://localhost:8000**.

### 3. Auto-load a model at startup (optional)

If you already have a trained model, mount it and set `MODEL_PATH`:

```bash
docker run -p 8000:8000 -v $(pwd)/models:/models -e MODEL_PATH=/models/cnn_model.pt fashion-cnn-server
```

## API Endpoints

### `GET /health`

Check if the server is running and which device is in use.

```bash
curl http://localhost:8000/health
# → {"status":"ok","device":"cpu"}
```

### `GET /classes`

List the 10 FashionMNIST class labels.

```bash
curl http://localhost:8000/classes
```

### `POST /upload-model`

Upload a trained PyTorch model file (`.pt` or `.pth`). Supports both:
- **Full model**: saved with `torch.save(model, 'model.pt')`
- **State dict**: saved with `torch.save(model.state_dict(), 'model.pt')`

```bash
curl -X POST http://localhost:8000/upload-model \
  -F "file=@cnn_model.pt"
# → {"message":"Model loaded successfully","device":"cpu"}
```

### `POST /predict`

Upload a 28×28 grayscale FashionMNIST image and get a prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.png"
# → {
#     "predicted_class": "Ankle boot",
#     "predicted_index": 9,
#     "probabilities": {
#       "T-shirt/top": 0.0012,
#       "Trouser": 0.0008,
#       ...
#       "Ankle boot": 0.9783
#     }
#   }
```

### `POST /predict-batch`

Upload multiple images in one request for batch inference.

```bash
curl -X POST http://localhost:8000/predict-batch \
  -F "files=@img1.png" \
  -F "files=@img2.png"
# → {"predictions": [{"filename":"img1.png","predicted_class":"Coat","predicted_index":4}, ...]}
```

## Interactive API Docs

Once the server is running, open the auto-generated Swagger UI:

👉 **http://localhost:8000/docs**

You can upload models and test predictions directly from the browser.

## Local Development (without Docker)

```bash
# Create conda environment
conda env create -f enviroment.yml
conda activate pytorch

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Or with pip:

```bash
pip install fastapi uvicorn[standard] python-multipart pillow torch torchvision
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Saving Your Trained Model

After training your CNN model (e.g. in `CNN.py`), export it so the server can load it:

```python
import torch
from CNN import YourCNN  # or define FashionCNN matching server.py

model = FashionCNN(num_classes=10)
# ... train your model ...
torch.save(model.state_dict(), "cnn_model.pt")
```

Then upload `cnn_model.pt` to the server via `POST /upload-model`.

## Model Architecture

The server expects a CNN with this architecture (defined in `server.py`):

| Layer       | Details                          |
|-------------|----------------------------------|
| Conv2d      | 1 → 32, kernel=3, padding=1      |
| BatchNorm2d | 32                               |
| MaxPool2d   | 2×2                              |
| Conv2d      | 32 → 64, kernel=3, padding=1     |
| BatchNorm2d | 64                               |
| MaxPool2d   | 2×2                              |
| Dropout     | 0.25                             |
| Linear      | 64×7×7 → 128                     |
| Dropout     | 0.25                             |
| Linear      | 128 → 10                         |

Input: `(1, 28, 28)` grayscale image. Output: 10 class logits.

## Project Files

| File               | Purpose                                      |
|--------------------|----------------------------------------------|
| `server.py`        | FastAPI server (single-file application)     |
| `Dockerfile`       | Docker container definition                  |
| `enviroment.yml`   | Conda environment specification              |
| `data_loader.py`   | FashionMNIST data loading & augmentation     |
| `CNN.py`           | CNN model training script (to be completed)  |
| `DNN.py`           | DNN model training script (to be completed)  |
