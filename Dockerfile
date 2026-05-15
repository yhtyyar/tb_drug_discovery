# Dockerfile for TB Drug Discovery API
# Provides production-ready deployment container

FROM python:3.10-slim

# System dependencies for RDKit and scientific computing
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn[standard] fastapi

# Copy application code
COPY src/ src/
COPY config/ config/
COPY models/ models/

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from api.app import app; print('OK')" || exit 1

# Expose API port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
