# FashionMNIST CNN Model Server

FastAPI server for FashionMNIST image classification with PyTorch CNN. Accepts JSON pixel data for single and batch prediction.

## Quick Start

```bash
# Create conda environment
conda env create -f environment.yml
conda activate pytorch

# Open the report notebook and run cells step by step
jupyter notebook report.ipynb
```

## API Endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
# → {"status":"ok","device":"cpu"}
```

### `GET /classes`

```bash
curl http://localhost:8000/classes
# → {"classes":["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]}
```

### `POST /predict-json`

Send JSON pixel data for single or batch prediction.

**Single prediction** — flat 784 array (28x28) or 2D 28x28 array:

```bash
curl -X POST http://localhost:8000/predict-json \
  -H "Content-Type: application/json" \
  -d '{"image": [0,0,0, ... 784 values ...]}'
# → {"predicted_class":"Ankle boot","predicted_index":9,"probabilities":{...}}
```

**Batch prediction** — multiple images:

```bash
curl -X POST http://localhost:8000/predict-json \
  -H "Content-Type: application/json" \
  -d '{"images": [[784 values], [784 values]]}'
# → {"predictions":[{"index":0,"predicted_class":"Coat","predicted_index":4}, ...]}
```

## Saving Your Trained Model

After training, save the state_dict so the server can load it:

```python
import torch
from cnn import CNN

model = CNN()
# ... train your model ...
torch.save(model.state_dict(), "cnn_model.pt")
```

## Model Architecture

The server expects a CNN with this architecture (defined in `cnn.py`):

| Layer       | Details                          |
|-------------|----------------------------------|
| Conv2d      | 1 → 6, kernel=5                  |
| ReLU        |                                  |
| MaxPool2d   | 2×2                              |
| Conv2d      | 6 → 16, kernel=5                 |
| ReLU        |                                  |
| MaxPool2d   | 2×2                              |
| Flatten     | 16×4×4 = 256                     |
| Linear      | 256 → 120                        |
| ReLU        |                                  |
| Linear      | 120 → 84                         |
| ReLU        |                                  |
| Linear      | 84 → 10                          |

Input: `(1, 28, 28)` grayscale image. Output: 10 class logits.

## Project Files

| File               | Purpose                                      |
|--------------------|----------------------------------------------|
| `server.py`        | FastAPI server (single-file application)     |
| `Dockerfile`       | Docker container definition                  |
| `environment.yml`  | Conda environment specification              |
| `data_loader.py`   | FashionMNIST data loading & augmentation     |
| `cnn.py`           | CNN model class and training script          |
| `dnn.py`           | DNN model class and training script          |
| `report.ipynb`     | Training, evaluation, and server test        |
