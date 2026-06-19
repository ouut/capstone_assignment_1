# FashionMNIST CNN Model Server

FastAPI server for FashionMNIST image classification with PyTorch CNN. Accepts JSON pixel data for single and batch prediction.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn[standard] pillow torch torchvision numpy

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server is live at **http://localhost:8000**。打开 **http://localhost:8000/docs** 查看 Swagger UI。

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

传入 JSON 像素数据，支持单张和批量预测。

**单张预测** — 传入 784 个像素值（28x28 拉平）或 28x28 二维数组：

```bash
curl -X POST http://localhost:8000/predict-json \
  -H "Content-Type: application/json" \
  -d '{"image": [0,0,0, ... 784 values ...]}'
# → {"predicted_class":"Ankle boot","predicted_index":9,"probabilities":{...}}
```

**批量预测** — 传入多个图像：

```bash
curl -X POST http://localhost:8000/predict-json \
  -H "Content-Type: application/json" \
  -d '{"images": [[784 values], [784 values]]}'
# → {"predictions":[{"index":0,"predicted_class":"Coat","predicted_index":4}, ...]}
```

## Saving Your Trained Model

训练完 CNN 模型后，保存 state_dict，服务器启动时自动加载：

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
| `report.ipynb`     | Full training, evaluation, and Docker test   |
