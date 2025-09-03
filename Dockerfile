# --- Builder Stage ---
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential git

# Set up working directory
WORKDIR /app

# Copy requirements and install with selective filtering
COPY requirements.txt .

# Install core packages with EXACT versions matching working venv
RUN pip install --no-cache-dir --prefix="/install" psutil && \
    pip install --no-cache-dir --prefix="/install" torch==2.6.0+cu126 torchaudio==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir --prefix="/install" ctranslate2==4.4.0 && \
    pip install --no-cache-dir --prefix="/install" whisperx==3.4.2

# Install remaining packages including text processing libraries and pyannote.audio
# Includes: nemo-text-processing (with pynini/OpenFST), contractions, pyannote.audio, and other dependencies
# Note: nemo-text-processing works on Linux x86_64 with existing build-essential
RUN grep -v "profanity" requirements.txt | grep -v "cu11" | grep -v "^torch" | grep -v "^ctranslate2" | grep -v "whisperx" > requirements_filtered.txt && \
    pip install --no-cache-dir --prefix="/install" -r requirements_filtered.txt

# --- Final Stage ---  
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/install/lib/python3.12/site-packages"

# PyTorch CUDA library isolation for optimal performance
ENV LD_LIBRARY_PATH="/install/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/install/lib/python3.12/site-packages/nvidia/cudnn/lib:/install/lib/python3.12/site-packages/nvidia/cublas/lib:/install/lib/python3.12/site-packages/nvidia/cufft/lib:/install/lib/python3.12/site-packages/nvidia/curand/lib:/install/lib/python3.12/site-packages/nvidia/cusolver/lib:/install/lib/python3.12/site-packages/nvidia/cusparse/lib"

# Install system dependencies including Python 3.12
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN apt-get update && apt-get install -y \
    software-properties-common \
    ffmpeg \
    libsndfile1 \
    python3-pip \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y python3.12 python3.12-dev python3.12-venv \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && rm -rf /var/lib/apt/lists/*

# Enhanced WhisperX Configuration (Customer Feedback Optimized)
ENV WHISPERX_COMPUTE_TYPE=float32
ENV WHISPERX_BATCH_SIZE=8
ENV WHISPERX_CHAR_ALIGN=true
ENV MAX_CONCURRENT_REQUESTS=3
ENV MEMORY_PER_REQUEST_GB=6.0

# Enhanced VAD Configuration (addressing customer feedback)
ENV WHISPERX_CHUNK_LENGTH=15
ENV VAD_ONSET=0.35
ENV VAD_OFFSET=0.25
ENV MIN_SEGMENT_LENGTH=0.5
ENV MAX_SEGMENT_LENGTH=30.0
ENV SPEECH_THRESHOLD=0.6

# Speaker Attribution Improvements (Optimized Diarization System)
ENV PYANNOTE_CLUSTERING_THRESHOLD=0.7
ENV PYANNOTE_SEGMENTATION_THRESHOLD=0.45
ENV SPEAKER_CONFIDENCE_THRESHOLD=0.6
ENV MIN_SPEAKER_DURATION=3.0
ENV SPEAKER_SMOOTHING_ENABLED=true
ENV MIN_SWITCH_DURATION=2.0
ENV VAD_VALIDATION_ENABLED=false

# WhisperX Core Settings
ENV ALIGN_MODEL=WAV2VEC2_ASR_LARGE_LV60K_960H
ENV INTERPOLATE_METHOD=linear
ENV SEGMENT_RESOLUTION=sentence
ENV ENABLE_REQUEST_QUEUING=true
ENV QUEUE_TIMEOUT=300

# Text Processing Enhancements
ENV TEXT_NORMALIZATION_MODE=enhanced

# Set up working directory and user
WORKDIR /app
RUN useradd -m appuser
USER appuser

# Copy application and dependencies from builder
COPY --from=builder /install /install
COPY python_server.py .
COPY env.example .

# Expose port and run application
EXPOSE 3333
CMD ["python3", "python_server.py"] 