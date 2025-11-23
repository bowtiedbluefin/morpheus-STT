# WhisperX API Parameters Reference

## Complete List of Available Parameters

### Input Source Parameters (choose ONE)
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `file` | File | Audio file to upload and transcribe | `audio.wav` |
| `storage_key` | String | Storage object key for existing file | "uploads/audio_123.wav" |
| `s3_presigned_url` | String | Pre-signed URL to download audio file from S3 | "https://bucket.s3.amazonaws.com/file?..." |

**Note**: Exactly one of these input sources must be provided. The API will fail if none are provided or if multiple are provided.

### Core Parameters
| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `language` | String | "en" | Language code (ISO-639-1) | "en", "es", "fr" |
| `response_format` | String | "json" | Output format | "json", "verbose_json", "text", "srt", "vtt" |
| `timestamp_granularities` | String | "segment" | Timestamp granularity (comma-separated for multiple) | "segment", "word", "segment,word" |

### Output Storage Parameters
| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `stored_output` | Boolean | false | Enable result storage | true, false |
| `upload_presigned_url` | String | None | Pre-signed upload URL to save results to your own S3 bucket | "https://bucket.s3.amazonaws.com/upload?..." |
| `output_key` | String | None | Custom storage object key (auto-generated if not provided) | "my-results/transcript.json" |

**Output Logic**:
- If `stored_output` is `false`: Results returned in API response only
- If `stored_output` is `true`:
  - If `upload_presigned_url` is provided: Upload to user's S3 bucket
  - Otherwise: Upload to server's downloads bucket

### WhisperX Enhanced Parameters
| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `output_content` | String | "both" | Response content control | "text_only", "timestamps_only", "both", "metadata_only" |
| `enable_diarization` | Boolean | **true** | Enable speaker identification | true, false |
| `prompt` | String | "" | Optional text prompt to guide transcription | "John Smith, Microsoft, API" |

### Fine-Tuning Parameters
| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `batch_size` | Integer | 16 | Batch size for processing (higher = faster but more memory) | 1-32 |
| `clustering_threshold` | Float | 0.7 | Speaker clustering threshold (lower = more speakers) | 0.5-1.0 |
| `segmentation_threshold` | Float | 0.45 | Voice activity detection threshold | 0.1-0.9 |
| `min_speaker_duration` | Float | 3.0 | Minimum speaking time per speaker (seconds) | 1.0-10.0 |
| `speaker_confidence_threshold` | Float | 0.6 | Minimum confidence for speaker assignment | 0.1-1.0 |
| `speaker_smoothing_enabled` | Boolean | true | Enable speaker transition smoothing | true, false |
| `min_switch_duration` | Float | 2.0 | Minimum time between speaker switches (seconds) | 0.5-5.0 |
| `vad_validation_enabled` | Boolean | false | Enable Voice Activity Detection validation (experimental) | true, false |
| `optimized_alignment` | Boolean | true | Use WAV2VEC2 alignment (true) or Whisper alignment (false) | true, false |

**Note**: `model` and `temperature` parameters are handled gracefully if sent by legacy clients but are not used.

## Parameter Details

### `response_format` Options
- **`json`**: Simple JSON with text and segments/words
- **`verbose_json`**: OpenAI-compatible format with metadata (task, language, duration)
- **`text`**: Plain text output only
- **`srt`**: SubRip subtitle format
- **`vtt`**: WebVTT subtitle format

### `timestamp_granularities` Options
- **`segment`**: Sentence/phrase level timestamps
- **`word`**: Individual word timestamps with confidence scores
- **`segment,word`**: Both segment and word-level timestamps (comma-separated)

### `output_content` Options
- **`text_only`**: Only transcribed text, no timestamps
- **`timestamps_only`**: Only timing information, no text content
- **`both`**: Separate text-only section AND timestamped results (default)
- **`metadata_only`**: Only processing information and speaker metadata

### `language` Codes (ISO-639-1)
Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ar` - Arabic
- `hi` - Hindi

## Response Formats Examples

### JSON Response
```json
{
  "text": "Hello, this is a transcription.",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a transcription."
    }
  ],
  "words": [
    {
      "start": 0.0,
      "end": 0.5,
      "text": "Hello",
      "decorated_text": "Hello",
      "word_prob": 0.95,
      "speaker": null
    },
    {
      "start": 0.5,
      "end": 1.2,
      "text": "this",
      "decorated_text": "this", 
      "word_prob": 0.98,
      "speaker": null
    }
  ]
}
```

### JSON with Diarization
```json
{
  "text": "Hello there. How are you today?",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello there.",
      "speaker": "SPEAKER_00"
    },
    {
      "start": 2.5,
      "end": 4.0,
      "text": "How are you today?",
      "speaker": "SPEAKER_01"
    }
  ],
  "speakers": 2,
  "words": [
    {
      "start": 0.0,
      "end": 0.5,
      "text": "Hello",
      "decorated_text": "Hello",
      "word_prob": 0.95,
      "speaker": "SPEAKER_00"
    },
    {
      "start": 0.6,
      "end": 1.1,
      "text": "there.",
      "decorated_text": "there.",
      "word_prob": 0.92,
      "speaker": "SPEAKER_00"
    },
    {
      "start": 2.5,
      "end": 2.8,
      "text": "How",
      "decorated_text": "How",
      "word_prob": 0.98,
      "speaker": "SPEAKER_01"
    }
  ]
}
```

### Verbose JSON Response
```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 44.08,
  "text": "Hello, this is a transcription.",
  "segments": [...],
  "speakers": 2
}
```

### Output Content Format Examples

#### text_only Format
```json
{
  "text": "Hello there. How are you today?"
}
```

#### timestamps_only Format
```json
{
  "status": "success",
  "result": {
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "speaker": "SPEAKER_00"
      },
      {
        "start": 2.5,
        "end": 4.0,
        "speaker": "SPEAKER_01"
      }
    ],
    "words": [
      {
        "start": 0.0,
        "end": 0.5,
        "word_prob": 0.95,
        "speaker": "SPEAKER_00"
      }
    ]
  },
  "processing_info": {...}
}
```

#### both Format (Default)
```json
{
  "status": "success",
  "text": "Hello there. How are you today?",
  "result": {
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Hello there.",
        "speaker": "SPEAKER_00"
      }
    ],
    "words": [...]
  },
  "processing_info": {...}
}
```

#### metadata_only Format
```json
{
  "status": "success",
  "processing_info": {
    "device": "cuda",
    "language_detected": "en",
    "total_speakers": 2,
    "confident_speakers": 2,
    "timestamp_granularities": ["segment"],
    "diarization_enabled": true
  },
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

## cURL Examples

### Basic Usage
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav"
```

### With All Core Parameters
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities=word" \
  -F "output_content=both" \
  -F "enable_diarization=true" \
  -F "prompt=Technical meeting with John Smith"
```

### With Fine-Tuning Parameters
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  -F "batch_size=16" \
  -F "clustering_threshold=0.7" \
  -F "segmentation_threshold=0.45" \
  -F "min_speaker_duration=3.0" \
  -F "speaker_confidence_threshold=0.6" \
  -F "speaker_smoothing_enabled=true" \
  -F "min_switch_duration=2.0" \
  -F "vad_validation_enabled=false" \
  -F "optimized_alignment=true"
```

### Text Only Output
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

### SRT Subtitles with Speaker Labels
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=srt" \
  -F "enable_diarization=true"
```

## Testing the API

### Health Check
```bash
curl http://localhost:3333/health
```

### Swagger UI
Visit: `http://localhost:3333/docs`

### Simple Test
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@test_audio.wav" \
  -F "response_format=json"
```

---

## Advanced Diarization Parameters

The system now includes state-of-the-art speaker diarization capabilities with configurable parameters for optimal accuracy.

### API Parameter Reference

All fine-tuning parameters listed above are available as API parameters. You can also set them as environment variables for server-wide defaults:

| API Parameter | Environment Variable | Type | Range | Default | Description |
|---------------|---------------------|------|-------|---------|-------------|
| `clustering_threshold` | `PYANNOTE_CLUSTERING_THRESHOLD` | float | 0.3-1.0 | 0.7 | Controls speaker clustering aggressiveness. Higher = fewer speakers |
| `segmentation_threshold` | `PYANNOTE_SEGMENTATION_THRESHOLD` | float | 0.1-1.0 | 0.45 | Speech detection sensitivity for segmentation |
| `speaker_confidence_threshold` | `SPEAKER_CONFIDENCE_THRESHOLD` | float | 0.0-1.0 | 0.6 | Minimum confidence for speaker assignment (SR-TH) |
| `min_speaker_duration` | `MIN_SPEAKER_DURATION` | float | 1.0-10.0 | 3.0 | Minimum speaking time for valid speaker |
| `speaker_smoothing_enabled` | `SPEAKER_SMOOTHING_ENABLED` | boolean | - | true | Reduce rapid speaker switches (A→B→A patterns) |
| `min_switch_duration` | `MIN_SWITCH_DURATION` | float | 0.5-5.0 | 2.0 | Minimum seconds between speaker changes |
| `vad_validation_enabled` | `VAD_VALIDATION_ENABLED` | boolean | - | false | Cross-validate with Voice Activity Detection |

### Enhanced Response Format

The diarization system returns enhanced speaker attribution with word-level confidence scores:

```json
{
  "status": "success",
  "result": {
    "segments": [
      {
        "id": 0,
        "seek": 0,
        "start": 0.5,
        "end": 3.2,
        "text": "Hello everyone, welcome to our meeting",
        "speaker": "SPEAKER_01",
        "speaker_confidence": 0.87,
        "tokens": [1234, 5678, 9012],
        "temperature": 0.0,
        "avg_logprob": -0.12,
        "compression_ratio": 2.4,
        "no_speech_prob": 0.01,
        "confidence": 0.95,
        "words": [
          {
            "word": "Hello",
            "start": 0.5,
            "end": 0.8,
            "score": 0.99,
            "speaker": "SPEAKER_01",
            "speaker_confidence": 0.91
          },
          {
            "word": "everyone",
            "start": 0.9,
            "end": 1.4,
            "score": 0.97,
            "speaker": "SPEAKER_01", 
            "speaker_confidence": 0.89
          }
        ]
      },
      {
        "id": 1,
        "seek": 320,
        "start": 3.5,
        "end": 6.1,
        "text": "Thank you for joining us today",
        "speaker": "SPEAKER_02",
        "speaker_confidence": 0.94,
        "words": [
          {
            "word": "Thank",
            "start": 3.5,
            "end": 3.8,
            "score": 0.98,
            "speaker": "SPEAKER_02",
            "speaker_confidence": 0.96
          }
        ]
      }
    ]
  },
  "processing_info": {
    "device": "cuda",
    "language_detected": "en",
    "audio_duration": 300.5,
    "total_speakers": 3,
    "confident_speakers": 3,
    "optimization_layers_applied": 6,
    "processing_time": 32.1,
    "gpu_memory_used": "1.2GB"
  }
}
```

### New Response Fields

#### Segment-Level Speaker Attribution
- `speaker`: Assigned speaker ID (e.g., "SPEAKER_01", "SPEAKER_02")
- `speaker_confidence`: Confidence score for speaker assignment (0.0-1.0)

#### Word-Level Speaker Attribution  
- `speaker`: Speaker ID for individual words
- `speaker_confidence`: Word-level speaker confidence score

#### Processing Information
- `total_speakers`: Total number of speakers detected
- `confident_speakers`: Number of speakers meeting confidence threshold
- `optimization_layers_applied`: Number of optimization layers processed
- `gpu_memory_used`: GPU memory consumption during processing

### Parameter Tuning Guidelines

#### Over-Detection Issues (Too Many Speakers)
**Via API Parameters:**
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  -F "clustering_threshold=0.8" \
  -F "speaker_confidence_threshold=0.8" \
  -F "min_speaker_duration=5.0"
```

**Via Environment Variables (server-wide defaults):**
```bash
export PYANNOTE_CLUSTERING_THRESHOLD=0.8    # Increase from 0.7
export SPEAKER_CONFIDENCE_THRESHOLD=0.8     # Increase from 0.6  
export MIN_SPEAKER_DURATION=5.0             # Increase from 3.0
```

#### Under-Detection Issues (Too Few Speakers)
**Via API Parameters:**
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  -F "clustering_threshold=0.5" \
  -F "speaker_confidence_threshold=0.4" \
  -F "vad_validation_enabled=true"
```

**Via Environment Variables:**
```bash
export PYANNOTE_CLUSTERING_THRESHOLD=0.5    # Decrease from 0.7
export SPEAKER_CONFIDENCE_THRESHOLD=0.4     # Decrease from 0.6
export VAD_VALIDATION_ENABLED=true          # Enable additional validation
```

#### Performance Optimization
```bash
export WHISPERX_BATCH_SIZE=4                # Reduce for lower memory usage
export SPEAKER_SMOOTHING_ENABLED=true       # Enable for better quality
export CUDA_MEMORY_OPTIMIZATION=true        # Enable for stability
```

### Testing Diarization

Test the enhanced diarization capabilities:

```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@multi_speaker_audio.wav" \
  -F "enable_diarization=true" \
  -F "response_format=json"
```

The response will include detailed speaker attribution with confidence scores for production-quality speaker recognition. 