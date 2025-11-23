# --- Builder Stage ---
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential git

# Set up working directory
WORKDIR /app

# Copy requirements and install with selective filtering
COPY requirements.txt .

# Install core packages
RUN pip install --no-cache-dir --prefix="/install" psutil && \
    pip install --no-cache-dir --prefix="/install" torch==2.6.0+cu126 torchaudio==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir --prefix="/install" ctranslate2==4.4.0 && \
    pip install --no-cache-dir --prefix="/install" whisperx==3.4.2

# Install remaining packages including text processing libraries and pyannote.audio
RUN pip install --no-cache-dir --prefix="/install" "transformers>=4.54.0" && \
    grep -v "profanity" requirements.txt | grep -v "cu11" | grep -v "^torch" | grep -v "^ctranslate2" | grep -v "whisperx" | grep -v "^tokenizers" | grep -v "^transformers" | grep -v "^av" > requirements_filtered.txt && \
    pip install --no-cache-dir --prefix="/install" --upgrade-strategy only-if-needed -r requirements_filtered.txt

# --- Final Stage ---  
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/install/lib/python3.12/site-packages"

# =============================================================================
# CUDA & GPU CONFIGURATION
# =============================================================================
# PyTorch CUDA library isolation 
ENV LD_LIBRARY_PATH="/install/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/install/lib/python3.12/site-packages/nvidia/cudnn/lib:/install/lib/python3.12/site-packages/nvidia/cublas/lib:/install/lib/python3.12/site-packages/nvidia/cufft/lib:/install/lib/python3.12/site-packages/nvidia/curand/lib:/install/lib/python3.12/site-packages/nvidia/cusolver/lib:/install/lib/python3.12/site-packages/nvidia/cusparse/lib"

# CUDA initialization and compatibility settings (critical for H100)
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
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

# =============================================================================
# WHISPERX CORE SETTINGS
# =============================================================================
# Batch size for processing (higher = faster but more memory)
ENV WHISPERX_BATCH_SIZE=16

# Use WAV2VEC2 alignment (true) or Whisper built-in alignment (false)
ENV USE_OPTIMIZED_ALIGNMENT=true

# Alignment Model Selection (optional - defaults to WhisperX's default if not set)
# Options: WAV2VEC2_ASR_BASE_960H (~360MB) or WAV2VEC2_ASR_LARGE_LV60K_960H (~1.2GB)
ENV ALIGN_MODEL=WAV2VEC2_ASR_LARGE_LV60K_960H

# =============================================================================
# DIARIZATION PARAMETERS
# =============================================================================
# Speaker clustering threshold (0.5-1.0, lower = more speakers detected)
ENV PYANNOTE_CLUSTERING_THRESHOLD=0.7

# Voice activity detection threshold (0.1-0.9)
ENV PYANNOTE_SEGMENTATION_THRESHOLD=0.45

# Minimum speaking time per speaker (seconds)
ENV MIN_SPEAKER_DURATION=3.0

# Minimum confidence for speaker assignment (0.1-1.0)
ENV SPEAKER_CONFIDENCE_THRESHOLD=0.6

# Enable speaker transition smoothing
ENV SPEAKER_SMOOTHING_ENABLED=true

# Minimum time between speaker switches (seconds)
ENV MIN_SWITCH_DURATION=2.0

# Enable Voice Activity Detection validation (experimental)
ENV VAD_VALIDATION_ENABLED=false

# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================
# Default storage buckets for uploads and results
ENV DEFAULT_UPLOAD_BUCKET=default-uploads
ENV DEFAULT_RESULTS_BUCKET=default-results

# =============================================================================
# CONCURRENCY & TIMEOUT CONFIGURATION
# =============================================================================
# Maximum concurrent GPU transcription requests
# Adjust based on GPU: RTX 3090=4, A100=8, H100=12-16
ENV MAX_CONCURRENT_REQUESTS=4

# Upload timeout for S3 presigned URL uploads (seconds)
ENV UPLOAD_TIMEOUT_SECONDS=3600

# Maximum time a job can run before being abandoned (seconds)
ENV JOB_PROCESSING_TIMEOUT_SECONDS=86400

# How long to keep job results available for retrieval (seconds)
ENV JOB_RESULT_RETENTION_SECONDS=86400

# Local directory for storing job results
ENV JOB_RESULTS_DIR=/tmp/whisper-job-results

# =============================================================================
# APPLICATION SETUP
# =============================================================================
WORKDIR /app
RUN useradd -m appuser
USER appuser

# Copy application and dependencies from builder
COPY --from=builder /install /install
COPY python_server.py .
COPY env.example .
COPY docker_entrypoint.sh .

# Make entrypoint executable
USER root
RUN chmod +x docker_entrypoint.sh
USER appuser

# Expose port and run application
EXPOSE 3333
ENTRYPOINT ["./docker_entrypoint.sh"]
