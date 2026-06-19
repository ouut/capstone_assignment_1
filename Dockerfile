# ============================================================
# FashionMNIST Prediction Server - Conda Docker
# ============================================================
FROM continuumio/miniconda3:latest

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libjpeg62-turbo-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Conda env from environment.yml ----
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# ---- App code ----
COPY server.py .
COPY cnn.py .
COPY dnn.py .

# ---- Trained model (baked into image) ----
COPY models/best_model.pt /app/models/best_model.pt

# ---- Env setup ----
ENV PATH=/opt/conda/envs/pytorch/bin:$PATH
ENV CONDA_DEFAULT_ENV=pytorch
ENV MODEL_PATH=/app/models/best_model.pt

RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
