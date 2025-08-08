# WhisperX API Parameters Reference

## Complete List of Available Parameters

### Required Parameters
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `file` | File | Audio file to transcribe | `audio.wav` |

### Core Parameters
| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `model` | String | "whisper-1" | Model identifier (ignored - always uses large-v3) | "whisper-1", "gpt-4" |
| `language` | String | None (auto-detect) | Language code (ISO-639-1) | "en", "es", "fr" |
| `response_format` | String | "json" | Output format | "json", "verbose_json", "text", "srt", "vtt" |
| `timestamp_granularities[]` | Array | ["segment"] | Timestamp detail level | ["segment"], ["word"], ["segment", "word"] |

### WhisperX Enhanced Parameters
| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `output_content` | String | "both" | Response content control | "text_only", "timestamps_only", "both" |
| `enable_diarization` | Boolean | false | Enable speaker identification | true, false |
| `min_speakers` | Integer | None | Minimum speakers for diarization | 2 |
| `max_speakers` | Integer | None | Maximum speakers for diarization | 5 |


### Legacy Parameters (Present but NOT used by WhisperX)
| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `prompt` | String | None | Initial prompt text | ⚠️ Accepted but ignored |
| `temperature` | Float | 0.0 | Sampling temperature | ⚠️ Accepted but ignored |
| `enable_profanity_filter` | Boolean | false | Profanity filtering | ⚠️ Accepted but not implemented |

## Parameter Details

### `response_format` Options
- **`json`**: Simple JSON with text and segments/words
- **`verbose_json`**: OpenAI-compatible format with metadata (task, language, duration)
- **`text`**: Plain text output only
- **`srt`**: SubRip subtitle format
- **`vtt`**: WebVTT subtitle format

### `timestamp_granularities[]` Options
- **`segment`**: Sentence/phrase level timestamps
- **`word`**: Individual word timestamps with confidence scores
- **`["segment", "word"]`**: Both segment and word-level timestamps

### `output_content` Options
- **`text_only`**: Only transcribed text, no timestamps
- **`timestamps_only`**: Only timing information, no text
- **`both`**: Complete transcription with timestamps (default)

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

## cURL Examples

### Basic Usage
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav"
```

### With All Parameters
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -F "output_content=both" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=5" \
  -F "output_content=both"
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

## Requirements for Features

### Speaker Diarization
- **Hugging Face Token**: Required, free account
- **GPU Memory**: Additional ~2GB VRAM
- **Audio Quality**: Clear speech with distinguishable speakers
- **Audio Length**: Works best with 30+ seconds

### Profanity Filter
- **No additional requirements**
- **Language Support**: Primarily English
- **Performance**: Minimal impact

## Removed Parameters (No longer supported in WhisperX)

These parameters were available in the original Whisper implementation but are **NOT supported** in WhisperX:

- `beam_size` - Beam search size
- `vad_filter` - Voice Activity Detection filter
- `vad_threshold` - VAD sensitivity threshold
- `min_silence_duration_ms` - Minimum silence duration
- `condition_on_previous_text` - Text conditioning
- `best_of` - Number of candidates
- `patience` - Beam search patience
- `compression_ratio_threshold` - Compression ratio limit
- `log_prob_threshold` - Log probability threshold
- `no_speech_threshold` - No speech detection threshold

## Migration Notes

If you were using the original Whisper API with these parameters, they should be removed from your requests as they are no longer functional and may cause confusion.

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