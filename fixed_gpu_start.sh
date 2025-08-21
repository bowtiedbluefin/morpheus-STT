#!/bin/bash

# Fixed GPU WhisperX Server Startup Script
# This script sets the critical LD_LIBRARY_PATH to prioritize SYSTEM cuDNN over venv cuDNN

echo "🚀 Starting Fixed GPU WhisperX Diarization Server"
echo "Setting CUDA library paths to avoid cuDNN conflicts..."

# Activate virtual environment
source /home/kyle/Desktop/whisper/venv/bin/activate

# CRITICAL FIX: Prioritize system cuDNN (9.8.0) over venv cuDNN (9.5.1) to fix sublibrary mismatch
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/home/kyle/Desktop/whisper/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/kyle/Desktop/whisper/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/kyle/Desktop/whisper/venv/lib/python3.12/site-packages/nvidia/cufft/lib:/home/kyle/Desktop/whisper/venv/lib/python3.12/site-packages/nvidia/curand/lib:/home/kyle/Desktop/whisper/venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/kyle/Desktop/whisper/venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/kyle/Desktop/whisper/venv/lib/python3.12/site-packages/nvidia/nccl/lib"

# Set HUGGINGFACE_TOKEN if needed
export HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-""}

echo "LD_LIBRARY_PATH set to prioritize SYSTEM cuDNN 9.8.0 over venv cuDNN 9.5.1"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "cuDNN version: $(python -c 'import torch; print(torch.backends.cudnn.version())')"

# Start the GPU server
echo "Starting working_gpu_diarization_server.py on port 3337..."
python working_gpu_diarization_server.py