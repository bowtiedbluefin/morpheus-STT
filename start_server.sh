#!/bin/bash

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated."
else
    echo "Error: Virtual environment 'venv' not found."
    exit 1
fi

# Set the correct library path for cuDNN from within the venv
VENV_CUDNN_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$VENV_CUDNN_PATH:$LD_LIBRARY_PATH"

echo "Starting Enhanced Whisper API Server..."
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# Start the server
uvicorn python_server:app --host 0.0.0.0 --port 3333