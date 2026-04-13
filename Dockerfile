# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (faiss / pypdf / sentence-transformers need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (better layer cache)
COPY requirements.txt .
RUN pip install -r requirements.txt

# App source
COPY app ./app

# Local file storage mount point
RUN mkdir -p /tmp/ai-orchestrator

EXPOSE 8000

# NOTE: docker-compose will be authored separately by the user.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
