# =============================================================================
# Dockerfile — Zero-Trust Deepfake Voice Defense System
# =============================================================================
#
# CPU-only image (default):
#   docker build -t ztdvd .
#   docker run -p 8501:8501 ztdvd
#
# GPU-enabled image (NVIDIA CUDA 12.1):
#   Uncomment the FROM line below and comment out the python:3.11-slim line.
#   Requires nvidia-container-toolkit on the host and the --gpus flag at runtime.
#
# =============================================================================

# -- CPU base -----------------------------------------------------------------
FROM python:3.11-slim

# -- NVIDIA CUDA base (uncomment for GPU support) -----------------------------
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# RUN apt-get update && apt-get install -y python3.11 python3-pip \
#     && ln -s /usr/bin/python3.11 /usr/bin/python \
#     && ln -s /usr/bin/pip3 /usr/bin/pip

# =============================================================================
# System dependencies
# =============================================================================

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python dependencies
# =============================================================================

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Copy project source
# =============================================================================

COPY . .

# Install the project itself in editable mode so imports work
RUN pip install --no-cache-dir -e .

# =============================================================================
# Runtime configuration
# =============================================================================

# Streamlit port
EXPOSE 8501

# Streamlit configuration: disable telemetry & browser auto-open
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# =============================================================================
# Entrypoint
# =============================================================================

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
