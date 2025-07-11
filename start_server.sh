#!/bin/bash

# Activate virtual environment first to get VIRTUAL_ENV path
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Error: Virtual environment 'venv' not found."
    exit 1
fi

# Configure environment for cuDNN compatibility using the venv path
VENV_LIB_PATH="$VIRTUAL_ENV/lib/python*/site-packages/nvidia/cudnn/lib"
if [ -d "$(eval echo $VENV_LIB_PATH)" ]; then
    export LD_LIBRARY_PATH="$(eval echo $VENV_LIB_PATH):${LD_LIBRARY_PATH}"
    echo "cuDNN library path set from venv."
else
    echo "Warning: venv cuDNN path not found. Falling back to system paths."
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
fi

# Set other CUDA environment variables
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print environment info
echo "Starting Enhanced Whisper API Server..."
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Start the server
uvicorn python_server:app --host 0.0.0.0 --port 3333 