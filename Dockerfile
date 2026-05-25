# ─────────────────────────────────────────────────────────────
# Stage 1: Builder — install dependencies
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime — lean final image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="AgriAI Team"
LABEL description="AI Crop Recommendation System — K-Means ML"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app.py          .
COPY ml/             ./ml/
COPY templates/      ./templates/
COPY static/         ./static/
COPY monitoring/     ./monitoring/
COPY requirements.txt .

# Create non-root user for security
RUN useradd -m -u 1001 appuser && \
    mkdir -p data && \
    chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=false
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 5000

# Health check — calls /api/health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

# Entrypoint: generate dataset + train model if needed, then start gunicorn
CMD ["sh", "-c", "python -c \"from ml.dataset import generate_dataset; from ml.model import CropRecommender; import os; generate_dataset() if not os.path.exists('data/crop_data.csv') else None; CropRecommender().train() if not os.path.exists('data/kmeans_model.pkl') else None\" && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app"]
