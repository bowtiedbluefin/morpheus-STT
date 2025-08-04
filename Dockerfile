# --- Builder Stage ---
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential

# Set up working directory
WORKDIR /app

# Install Python dependencies - use official WhisperX installation
RUN pip install --no-cache-dir --prefix="/install" whisperx psutil fastapi uvicorn python-multipart

# --- Final Stage ---  
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/install/lib/python3.10/site-packages"

# CRITICAL: PyTorch CUDA library isolation to fix cuDNN version conflicts
ENV LD_LIBRARY_PATH="/install/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/install/lib/python3.10/site-packages/nvidia/cudnn/lib:/install/lib/python3.10/site-packages/nvidia/cublas/lib:/install/lib/python3.10/site-packages/nvidia/cufft/lib:/install/lib/python3.10/site-packages/nvidia/curand/lib:/install/lib/python3.10/site-packages/nvidia/cusolver/lib:/install/lib/python3.10/site-packages/nvidia/cusparse/lib"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# WhisperX Optimization 
ENV WHISPERX_COMPUTE_TYPE=float32
ENV WHISPERX_BATCH_SIZE=8
ENV WHISPERX_CHAR_ALIGN=true
ENV MAX_CONCURRENT_REQUESTS=4
ENV MEMORY_PER_REQUEST_GB=6.0
ENV WHISPERX_CHUNK_LENGTH=15
ENV VAD_ONSET=0.500
ENV VAD_OFFSET=0.200
ENV ALIGN_MODEL=WAV2VEC2_ASR_LARGE_LV60K_960H
ENV INTERPOLATE_METHOD=linear
ENV SEGMENT_RESOLUTION=sentence
ENV ENABLE_REQUEST_QUEUING=true
ENV QUEUE_TIMEOUT=300

# Set up working directory and user
WORKDIR /app
RUN useradd -m appuser
USER appuser

# Copy application and dependencies from builder
COPY --from=builder /install /install
COPY python_server.py .
COPY config_examples.env .

# Expose port and run application
EXPOSE 3333
CMD ["python3", "python_server.py"] 