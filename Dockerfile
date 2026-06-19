# ---- Conda-based PyTorch + FastAPI image ----
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install the conda environment from enviroment.yml
COPY enviroment.yml .
RUN conda env create -f enviroment.yml && conda clean -afy

# Copy application code (CNN.py = model class, server.py = FastAPI app)
COPY CNN.py server.py .

EXPOSE 8000

# Launch with conda run to activate the pytorch environment
CMD ["conda", "run", "--no-capture-output", "-n", "pytorch", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
