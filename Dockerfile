
FROM python:3.11-slim

# Application directory
WORKDIR /app

# System dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python requirements
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY api/ ./api/
COPY foodai/ ./foodai/
COPY config/ ./config/
COPY data/nutrition/ ./data/nutrition/
COPY frontend/ ./frontend/
COPY predict.py ./predict.py

# Model directory
RUN mkdir -p /app/models

# API port
EXPOSE 8000

# FastAPI production command
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
