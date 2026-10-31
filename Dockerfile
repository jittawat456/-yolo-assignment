FROM python:3.11-slim

# Keep Python output unbuffered and avoid writing bytecode into the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# libgl1/libglib2.0-0 are required by OpenCV (installed by Ultralytics);
# libgomp1 is required by PyTorch.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first so Ultralytics does not pull the large CUDA build.
COPY requirements.txt .
# hadolint ignore=DL3013
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code and both required weights.
COPY . .

# Run as an unprivileged user. The app writes nothing to disk; /tmp is used
# only for Ultralytics/Matplotlib configuration caches.
RUN groupadd --gid 10001 appuser \
    && useradd --create-home --uid 10001 --gid 10001 appuser \
    && mkdir -p /tmp/Ultralytics /tmp/matplotlib \
    && chown -R appuser:appuser /app /tmp/Ultralytics /tmp/matplotlib
USER 10001:10001

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','10000') + '/health', timeout=4)"]

# Render provides $PORT. One worker is intentional: both YOLO models are loaded
# once per process, so extra workers would multiply RAM usage.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --worker-class gthread --workers 1 --threads 4 --timeout 300 --access-logfile - --error-logfile - app:app"]
