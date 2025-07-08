# --- Builder Stage ---
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HUGGINGFACE_TOKEN=""

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Set up working directory and user
WORKDIR /app
RUN useradd -m appuser
USER appuser

# Copy application and dependencies from builder
COPY --from=builder /app /app
COPY python_server.py .
COPY .env .

# Set Hugging Face token from build argument
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Expose port and run application
EXPOSE 3333
CMD ["python3", "python_server.py"] 