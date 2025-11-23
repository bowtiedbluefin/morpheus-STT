#!/bin/bash
set -e

# Docker Entrypoint for WhisperX Server with GPU Initialization
# Handles CUDA initialization issues on high-end GPUs like H100

echo "=== WhisperX Server GPU Initialization ==="

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi found"
    
    # Display GPU information
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader
    
    # Check for any processes using GPU
    echo ""
    echo "Checking for GPU processes..."
    GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "")
    if [ -z "$GPU_PROCESSES" ]; then
        echo "No GPU processes detected - GPU is available"
    else
        echo "WARNING: GPU processes detected:"
        echo "$GPU_PROCESSES"
        echo "Attempting to clear GPU state..."
        
        # Try to reset GPU (requires sufficient permissions)
        nvidia-smi --gpu-reset 2>/dev/null || echo "Cannot reset GPU (insufficient permissions)"
    fi
    
    # Wait briefly to ensure GPU is ready (critical for H100)
    echo ""
    echo "Waiting for GPU initialization..."
    sleep 2
    
else
    echo "WARNING: nvidia-smi not found - running without GPU"
fi

# Display CUDA environment variables
echo ""
echo "CUDA Environment:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "  CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-<not set>}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<not set>}"

# Verify Python and dependencies
echo ""
echo "Python Environment:"
python3 --version
echo -n "PyTorch: "
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED"
echo -n "CUDA Available: "
python3 -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null || echo "ERROR"
echo -n "WhisperX: "
python3 -c "import whisperx; print('INSTALLED')" 2>/dev/null || echo "NOT INSTALLED"

echo ""
echo "=== Starting WhisperX Server ==="
echo ""

# Start the server
exec python3 python_server.py

