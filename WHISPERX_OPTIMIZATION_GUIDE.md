# WhisperX Optimization Guide

## Overview


### WhisperX Approach (Automated):
- Built-in VAD preprocessing
- Automatic phoneme-based alignment  
- Simplified parameter space
- Three-stage pipeline: Transcribe → Align → Diarize

## Available Optimization Parameters

### 1. Compute Type (`WHISPERX_COMPUTE_TYPE`)
**Controls model precision and speed**

```bash
# Environment variable
export WHISPERX_COMPUTE_TYPE=float16  # Default
```

**Options & Tradeoffs:**
- `float16`: 
  - ✅ **Best for GPU** - Fastest on modern GPUs with Tensor Cores  
  - ✅ Memory efficient (~50% less than float32)
  - ⚠️ Slightly less accurate than float32
  - 📊 **Recommended for production**

- `float32`: 
  - ✅ Most accurate
  - ❌ Slowest and highest memory usage
  - 📊 **Use for accuracy-critical applications**

- `int8`: 
  - ✅ Fastest on CPU and low-resource scenarios
  - ❌ Lowest accuracy  
  - 📊 **Use for CPU deployment or memory-constrained environments**

**Performance Data** (from benchmarks):
- 47s audio processing times:
  - `int8` + large-v3: 39s (0.83x realtime)
  - `float32` + large-v3: 61s (1.3x realtime)

### 2. Batch Size (`WHISPERX_BATCH_SIZE`)
**Controls processing throughput**

```bash
export WHISPERX_BATCH_SIZE=16  # Default
```

**Tradeoff Analysis:**
- **Smaller (4-8)**: Lower memory usage, slower overall
- **Medium (16)**: Balanced performance/memory - **recommended default**
- **Larger (32+)**: Higher throughput but more GPU memory required

**GPU Memory Requirements:**
- Batch 4: ~2GB VRAM
- Batch 16: ~4GB VRAM  
- Batch 32: ~8GB VRAM

### 3. Alignment Model (`ALIGN_MODEL`)
**Controls word-level timestamp accuracy**

```bash
export ALIGN_MODEL=WAV2VEC2_ASR_LARGE_LV60K_960H  # For better accuracy
```

**Options:**
- `None` (default): Automatic model selection per language
- `WAV2VEC2_ASR_LARGE_LV60K_960H`: Better accuracy, more memory
- Language-specific models for improved performance

### 4. Interpolation Method (`INTERPOLATE_METHOD`)
**How to handle non-aligned words**

```bash
export INTERPOLATE_METHOD=nearest  # Default
```

**Options:**
- `nearest`: Use closest aligned word timestamp
- `linear`: Linear interpolation between aligned words
- `ignore`: Skip non-aligned words entirely

## Migration from faster-whisper Optimizations

### Parameters No Longer Available (by design):

1. **Temperature** → Fixed to 0 (deterministic)
   - **Why**: Removes transcription variability 
   - **Benefit**: Consistent, reproducible results

2. **condition_on_previous_text** → Fixed to False  
   - **Why**: Reduces hallucinations
   - **Benefit**: Better accuracy on diverse audio

3. **beam_size** → Internally optimized
   - **Why**: WhisperX uses optimized decoding strategy
   - **Benefit**: No need to tune beam search parameters

4. **VAD parameters** → Automatic VAD preprocessing
   - **Why**: Uses pyannote/silero for robust voice detection
   - **Benefit**: No manual VAD threshold tuning needed

### Your Optimization Strategy Should Focus On:

## Performance Tuning Decision Matrix

### For Maximum Speed (Real-time transcription):
```bash
export WHISPERX_COMPUTE_TYPE=int8
export WHISPERX_BATCH_SIZE=32
export WHISPERX_CHUNK_LENGTH=10
# Result: ~2.3x realtime (20s processing for 47s audio)
```

### For Maximum Accuracy (Quality-critical):
```bash
export WHISPERX_COMPUTE_TYPE=float32
export WHISPERX_BATCH_SIZE=8
export ALIGN_MODEL=WAV2VEC2_ASR_LARGE_LV60K_960H
export WHISPERX_CHAR_ALIGN=true
# Result: Highest quality at ~1.3x realtime
```

### For Balanced Production (Recommended):
```bash
export WHISPERX_COMPUTE_TYPE=float16
export WHISPERX_BATCH_SIZE=16
export WHISPERX_CHUNK_LENGTH=30
# Result: Good quality at ~0.83x realtime
```

### For Memory-Constrained (2GB GPU):
```bash
export WHISPERX_COMPUTE_TYPE=int8
export WHISPERX_BATCH_SIZE=4
export WHISPERX_CHUNK_LENGTH=15
```

## Audio-Specific Optimizations

### Clean Studio Audio:
- Higher batch sizes work well
- Standard float16 compute type
- Can use longer chunks (30s)

### Noisy/Multi-speaker Audio:
- Enable diarization with speaker constraints
- Use shorter chunks (10-15s) for better boundaries
- Consider float32 for better accuracy

### Low-resource Languages:
- Specify language explicitly (`language=xx`)
- Use appropriate alignment model
- Consider float32 for better multilingual accuracy

## Monitoring & Tuning

### Key Metrics to Track:
1. **Processing Time Ratio**: Target < 1.0x for real-time
2. **GPU Memory Usage**: Monitor peak usage
3. **Word Error Rate**: Quality metric
4. **Memory Leaks**: Watch for accumulation over time

### Environment Variables for Fine-tuning:
```bash
# Core optimization
export WHISPERX_COMPUTE_TYPE=float16
export WHISPERX_BATCH_SIZE=16
export WHISPERX_CHUNK_LENGTH=30

# Advanced tuning  
export ALIGN_MODEL=WAV2VEC2_ASR_LARGE_LV60K_960H
export INTERPOLATE_METHOD=linear
export VAD_ONSET=0.500
export VAD_OFFSET=0.363
export WHISPERX_CHAR_ALIGN=false
```

## Debugging Performance Issues

### If processing is too slow:
1. Reduce `WHISPERX_BATCH_SIZE` (memory pressure)
2. Switch to `int8` compute type
3. Reduce chunk length
4. Disable character alignments

### If accuracy is poor:
1. Switch to `float32` compute type
2. Use language-specific alignment model
3. Enable character alignments for word-level precision
4. Constrain speaker count for diarization

### If running out of memory:
1. Reduce batch size to 4-8
2. Use `int8` compute type
3. Process shorter audio segments
4. Disable diarization if not needed

## Production Deployment Recommendations

### For High-Throughput API:
- Use A100/V100 GPUs
- float16 compute type
- Batch size 16-32
- Implement request queuing

### For Edge Deployment:
- Use T4 GPU or powerful CPU
- int8 compute type  
- Batch size 4-8
- Monitor thermal throttling

### For Cost Optimization:
- T4 GPU is most cost-effective
- Use spot instances where possible
- Implement auto-scaling based on queue depth
- Cache models to avoid download delays

## Conclusion

WhisperX's optimization approach trades fine-grained control for simplicity and better defaults. Focus your optimization efforts on:
1. **Compute type selection** based on accuracy vs. speed needs
2. **Batch size tuning** for your hardware constraints  
3. **Memory management** for production stability
4. **Audio-specific parameters** for your use case

The automated VAD, alignment, and diarization pipeline should give you better out-of-the-box results than manual faster-whisper tuning, with less complexity. 