#!/bin/bash

# WhisperX Diarization Server with GPU Optimization
# Includes cuDNN compatibility and GPU memory management
# CRITICAL FIX: Prioritizes system cuDNN over venv cuDNN to prevent sublibrary mismatch

echo "🚀 Starting WhisperX Diarization Server with GPU optimization..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Error: Virtual environment 'venv' not found"
    exit 1
fi

# CRITICAL GPU FIX: Set CUDA library paths to prevent cuDNN version conflicts
# Prioritizes system cuDNN (typically newer) over virtual environment cuDNN
echo "Setting CUDA library paths to prioritize SYSTEM cuDNN over venv cuDNN..."
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/curand/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusparse/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

# GPU memory optimization
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:512"}

# Load environment configuration if available
if [ -f ".env" ]; then
    source .env
    echo "Environment configuration loaded"
fi

# Set HuggingFace token for diarization
export HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-""}

# Enhanced system status check with cuDNN information
echo "System status:"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not available')"
echo "  cuDNN: $(python -c 'import torch; print(torch.backends.cudnn.version())' 2>/dev/null || echo 'Not available')"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi
echo "  LD_LIBRARY_PATH configured to prioritize system cuDNN"

# Start the server
echo "Starting server on port ${PORT:-3333}..."
uvicorn python_server:app --host 0.0.0.0 --port ${PORT:-3333}