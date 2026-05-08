FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV + MediaPipe runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Pre-download InsightFace model để build cache vào image
RUN python -c "from insightface.app import FaceAnalysis; \
    a=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
    a.prepare(ctx_id=-1, det_size=(640,640))"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
