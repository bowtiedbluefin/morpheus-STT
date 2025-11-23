#!/bin/bash
# Quick deployment script for Async Mode v4.6

set -e

echo "============================================================"
echo "WhisperX Async Mode v4.6 - Quick Deploy"
echo "============================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "python_server.py" ]; then
    echo "Error: python_server.py not found"
    echo "   Please run this script from the whisper directory"
    exit 1
fi

echo "Pre-flight checks..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

echo "Docker is running"

# Check if logged into Docker Hub
if ! docker info | grep -q "Username"; then
    echo "Not logged into Docker Hub"
    echo "   Run: docker login"
    read -p "   Press Enter after logging in..."
fi

echo "Docker Hub access ready"
echo ""

# Confirm deployment
echo "Ready to deploy v4.6 with async mode support"
echo ""
echo "This will:"
echo "  1. Build Docker image (5-10 min)"
echo "  2. Push to Docker Hub (5-15 min)"
echo "  3. You'll need to update RunPod manually"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

echo ""
echo "============================================================"
echo "STEP 1: Building Docker Image"
echo "============================================================"
echo ""

docker build -t kylecohen01/whisper-transcription-api:4.6 .

echo ""
echo "Build complete!"
echo ""
echo "============================================================"
echo "STEP 2: Pushing to Docker Hub"
echo "============================================================"
echo ""

docker push kylecohen01/whisper-transcription-api:4.6

echo ""
echo "Push complete!"
echo ""
echo "============================================================"
echo "STEP 3: Update RunPod (Manual)"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Go to your RunPod dashboard"
echo "2. Stop the current pod"
echo "3. Edit pod configuration"
echo "4. Change image to: kylecohen01/whisper-transcription-api:4.6"
echo "5. Start the pod"
echo ""
echo "OR use RunPod CLI:"
echo "  runpod update --image kylecohen01/whisper-transcription-api:4.6"
echo ""
echo "============================================================"
echo "STEP 4: Test Deployment"
echo "============================================================"
echo ""
echo "After RunPod update (wait ~2-3 min), run:"
echo "  python3 test_runpod_endpoint.py"
echo ""
echo "Expected output:"
echo "  ASYNC MODE IS ENABLED!"
echo ""
echo "============================================================"
echo "Docker image ready for deployment!"
echo "============================================================"
echo ""
echo "Image: kylecohen01/whisper-transcription-api:4.6"
echo "Status: Pushed to Docker Hub"
echo "Next: Update RunPod to use new image"
echo ""

