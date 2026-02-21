# ────────────────────────────────────────────────────────────────────
# Dockerfile
# Multi-Modal Diabetic Complication Predictor — Portfolio Project
# ────────────────────────────────────────────────────────────────────
# Build:   docker build -t diabeticeye-ai .
# Run app: docker run -p 8501:8501 diabeticeye-ai
# ────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps (OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Entry point: launch app
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
