# Deployment Configuration Updates

## Overview

Updated `Dockerfile` and `deploy-gpu.yaml` to reflect:
- **15x faster than realtime** performance (measured on RTX 3090)
- **cuDNN v9 compatibility** fixes  
- **Customer feedback optimizations** (speaker over-detection, punctuation, diarization)
- **Reduced resource requirements** due to improved performance

## Key Changes

### Dockerfile Updates

1. **Base Image**: `nvidia/cuda:12.4-cudnn9-runtime-ubuntu22.04` (was cudnn8)
2. **Python Version**: 3.12 (was 3.10)
3. **Dependencies**: Official WhisperX installation + required packages
4. **Critical Fix**: Added `LD_LIBRARY_PATH` for PyTorch CUDA library isolation
5. **Environment Variables**: All WhisperX optimization settings included

### deploy-gpu.yaml Updates

1. **Performance**: Updated timeouts for 15x faster processing
2. **Environment**: Complete WhisperX configuration included
3. **Resources**: Reduced CPU/RAM, prioritized GPU selection based on testing  
4. **Storage**: Increased for alignment models
5. **Pricing**: Reduced due to better performance per cost

## Building the Docker Image

```bash
# Build with updated Dockerfile
docker build -t whisperx-api:latest .

# Tag for deployment
docker tag whisperx-api:latest your-dockerhub-username/whisper-transcription-api:latest

# Push to registry
docker push your-dockerhub-username/whisper-transcription-api:latest
```

## Deployment Performance Expectations

### RTX 3090 (Tested)
- **Processing Speed**: 15x faster than realtime  
- **Example**: 10-minute audio = 39 seconds processing
- **Concurrent Requests**: 4 simultaneous
- **Throughput**: 3600+ minutes audio/hour

### RTX 4090 (Projected) 
- **Processing Speed**: ~20x faster than realtime
- **Concurrent Requests**: 5-6 simultaneous  
- **Throughput**: 4800+ minutes audio/hour

### H100 (Projected)
- **Processing Speed**: 25-30x faster than realtime
- **Concurrent Requests**: 15+ simultaneous
- **Throughput**: 22,500+ minutes audio/hour

## Environment Variables

The deployment includes customer feedback optimized settings:

```yaml
# Core Performance
WHISPERX_COMPUTE_TYPE=float32           # Max accuracy
WHISPERX_BATCH_SIZE=8                   # Balanced performance
MAX_CONCURRENT_REQUESTS=4               # RTX 3090 optimized

# Customer Feedback Fixes  
VAD_ONSET=0.500                         # Reduce speaker over-detection
VAD_OFFSET=0.200                        # Tighter speaker boundaries
WHISPERX_CHAR_ALIGN=true                # Better punctuation
SEGMENT_RESOLUTION=sentence             # Improved context
```

## Usage Notes

1. **Speaker Count**: Always use `min_speakers`/`max_speakers` parameters:
   ```bash
   curl -F "min_speakers=2" -F "max_speakers=2" ...
   ```

2. **Performance Monitoring**: Check real-time status:
   ```bash
   curl http://your-deployment-url/processing-status
   ```

3. **Memory**: GPU VRAM is the limiting factor, not system RAM

4. **Scaling**: With 15x realtime performance, fewer instances needed

## Cost Analysis

- **Better Performance/Cost**: 15x speed improvement reduces compute hours needed
- **Lower Pricing**: Reduced Akash deployment bid due to efficiency gains  
- **ROI**: Dramatically improved vs previous estimates

## Troubleshooting

If you encounter cuDNN issues:
1. Ensure using the updated `cudnn9` base image
2. Verify `LD_LIBRARY_PATH` is set correctly  
3. Check GPU compatibility (minimum: compute capability 7.0+)
4. Monitor logs for version mismatch warnings

The updated configuration addresses all discovered compatibility issues and optimizes for the measured 15x performance improvement. 