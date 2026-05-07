FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements-docker-mac.txt .
RUN grep -v "openai-whisper\|faster-whisper" requirements-docker-mac.txt \
    | pip install --no-cache-dir -r /dev/stdin

COPY . .

# Patch any whisper imports so app doesn't crash on startup
RUN find . -name "*.py" -exec \
    sed -i 's/^import whisper/try:\n    import whisper\nexcept ImportError:\n    whisper = None/g' {} \; 2>/dev/null || true
RUN find . -name "*.py" -exec \
    sed -i 's/^from faster_whisper/try:\n    from faster_whisper/g' {} \; 2>/dev/null || true

RUN pip install --no-cache-dir -e . 2>/dev/null || true

EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", "--server.port=8501"]
