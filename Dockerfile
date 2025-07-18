# --- Builder Stage ---
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

# --- Final Stage ---
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/install/lib/python3.10/site-packages"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Set up working directory and user
WORKDIR /app
RUN useradd -m appuser
USER appuser

# Copy application and dependencies from builder
COPY --from=builder /install /install
COPY python_server.py .

# Expose port and run application
EXPOSE 3333
CMD ["python3", "python_server.py"] 