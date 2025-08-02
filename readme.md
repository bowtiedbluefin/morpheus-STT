# WhisperX Transcription API

## Features

- **🆕 WhisperX Integration**: Enhanced transcription with improved accuracy and speed
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's Whisper API
- **Multiple Output Formats**: JSON, text, SRT, VTT with flexible content options
- **Word-Level Timestamps**: Precise word-level timing with confidence scores
- **🆕 Built-in Speaker Diarization**: Identify and label different speakers automatically
- **Legacy Compatibility**: Supports OpenAI API parameters for easy migration
- **GPU Acceleration**: CUDA support for faster processing
- **Multiple Languages**: Auto-detection and manual language specification

## What's New with WhisperX

WhisperX provides significant improvements over standard Whisper:
- **Better Accuracy**: Enhanced transcription quality
- **Word-Level Timestamps**: More precise timing information
- **Integrated Diarization**: Built-in speaker identification
- **Faster Processing**: Optimized for speed and efficiency

## Installation

### Prerequisites
1. Python 3.8+ installed
2. NVIDIA GPU with CUDA support (recommended)
3. Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository-url>
cd whisper
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Start the server:**
```bash
bash start_server.sh
```

The server will start on `http://localhost:3333`

### Optional: Speaker Diarization Setup

To enable speaker diarization, you'll need a Hugging Face token:

1. **Get Hugging Face Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with READ permission
   - Accept the user conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1

2. **Set Environment Variable**:
   ```bash
   # Create .env file
   echo "HUGGINGFACE_TOKEN=your_token_here" > .env
   
   # Or set directly in environment
   export HUGGINGFACE_TOKEN=your_token_here
   ```

### Model Download

**No manual model download required!** WhisperX models will be automatically downloaded on first run. The download may take a few minutes depending on your internet connection.

- **First run**: Models download automatically, may take 5-10 minutes
- **Subsequent runs**: Uses cached models, starts immediately

## API Usage

### Endpoints

- **GET** `/` - Welcome message and basic info
- **GET** `/health` - Health check with model status
- **POST** `/v1/audio/transcriptions` - OpenAI-compatible transcription endpoint (recommended)
- **POST** `/transcribe` - Alias to the main endpoint (same functionality)
- **GET** `/docs` - Interactive API documentation (Swagger UI)

### Using the API

#### 1. Web Interface (Easiest)
Visit `http://localhost:3333/docs` in your browser to use the interactive API documentation where you can upload files directly and test all parameters.

#### 2. cURL Command Examples

##### Basic Usage (OpenAI Compatible)
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio/file.wav"
```

##### 🆕 Speaker Diarization Examples
```bash
# Basic diarization
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  -F "response_format=json"

# Diarization with speaker constraints
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  -F "response_format=json"

# SRT format with speaker labels
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "enable_diarization=true" \
  -F "response_format=srt" \
  -o subtitles_with_speakers.srt
```

##### 🆕 Advanced Features Examples
```bash
# Multiple timestamp granularities
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "timestamp_granularities=word" \
  -F "timestamp_granularities=segment" \
  -F "response_format=json"

# Language-specific with diarization
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "enable_diarization=true" \
  -F "response_format=verbose_json"
```

##### All Available Options Example
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio/file.wav" \
  -F "language=en" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -F "output_content=both" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=5" \
  -o transcription_with_words.json
```

##### Different Response Formats
```bash
# JSON with segment timestamps (default)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=segment"

# JSON with word-level timestamps
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=word"

# Verbose JSON with metadata (OpenAI format)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"

# Plain text output
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=text"

# SRT subtitle format
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=srt"

# WebVTT subtitle format
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=vtt"
```

##### Output Content Options
```bash
# Text only (no timestamps)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=json" \
  -F "output_content=text_only"

# Timestamps only (no text)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=segment" \
  -F "output_content=timestamps_only"

# Both text and timestamps (default)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=word" \
  -F "output_content=both"
```

##### Language-Specific Transcription
```bash
# Spanish audio
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@spanish_audio.wav" \
  -F "language=es"

# French audio
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@french_audio.wav" \
  -F "language=fr"

# Auto-detect language (default)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@multilingual_audio.wav"
```

#### 3. Python requests (OpenAI Compatible)
```python
import requests

url = "http://localhost:3333/v1/audio/transcriptions"
files = {"file": open("audio.wav", "rb")}
data = {
    "language": "en",
    "response_format": "verbose_json",
    "timestamp_granularities[]": "word",
    # WhisperX features
    "enable_diarization": True,
    "min_speakers": 2,
    "max_speakers": 4,
    "output_content": "both"
}
response = requests.post(url, files=files, data=data)
result = response.json()
print(result["text"])
print(f"Speakers found: {result.get('speakers', 'N/A')}")
```

#### 4. JavaScript/Node.js (OpenAI Compatible)
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('audio.wav'));
form.append('language', 'en');
form.append('response_format', 'verbose_json');
form.append('timestamp_granularities[]', 'segment');
// WhisperX features
form.append('enable_diarization', 'true');
form.append('min_speakers', '2');
form.append('max_speakers', '4');
form.append('output_content', 'both');

axios.post('http://localhost:3333/v1/audio/transcriptions', form, {
  headers: form.getHeaders()
}).then(response => {
  console.log(response.data.text);
  console.log(`Speakers found: ${response.data.speakers || 'N/A'}`);
});
```

#### 5. OpenAI Python Client (Drop-in Replacement)
```python
# You can use the official OpenAI client by pointing it to your server
from openai import OpenAI

# Point to your local server
client = OpenAI(
    api_key="dummy-key",  # Not used but required
    base_url="http://localhost:3333"
)

# Use exactly like OpenAI API
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",  # Ignored - always uses large-v3
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    print(transcript.text)
```

## API Parameters

### Required Parameters
- **`file`**: Audio file to transcribe (mp3, wav, m4a, flac, etc.)

### Core Parameters
- **`language`**: Language code (ISO-639-1) like "en", "es", "fr" - improves accuracy and speed
- **`response_format`**: Output format - `json`, `text`, `srt`, `vtt`, `verbose_json`
- **`timestamp_granularities[]`**: Array or single value - `segment`, `word`

### WhisperX Enhanced Parameters
- **`output_content`**: Control response content: `text_only`, `timestamps_only`, or `both` (default: 'both')
- **`enable_diarization`**: Enable speaker diarization to identify who is speaking (default: False)
- **`min_speakers`**: Minimum number of speakers to detect for diarization (optional)
- **`max_speakers`**: Maximum number of speakers to detect for diarization (optional)


### Legacy Parameters (Present but not used by WhisperX)
- **`model`**: Model identifier (ignored - always uses WhisperX large-v3) - for OpenAI compatibility
- **`prompt`**: Initial prompt (accepted for compatibility but not used)
- **`temperature`**: Sampling temperature (accepted for compatibility but not used)
- **`enable_profanity_filter`**: Profanity filter (accepted for compatibility but not implemented)

### Response Formats

#### `json` (Simple JSON)
```json
{
  "text": "Hello, this is a transcription.",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a transcription.",
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "score": 0.95
        }
      ]
    }
  ]
}
```

#### 🆕 `json` with Speaker Diarization
```json
{
  "text": "Hello there. How are you today?",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello there.",
      "speaker": "SPEAKER_00",
      "words": [...]
    },
    {
      "start": 2.5,
      "end": 4.0,
      "text": "How are you today?",
      "speaker": "SPEAKER_01",
      "words": [...]
    }
  ],
  "speakers": 2
}
```

#### `verbose_json` (OpenAI Format with Metadata)
```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 44.08,
  "text": "Hello, this is a transcription.",
  "segments": [...]
}
```

#### `text` (Plain Text)
```
Hello, this is a transcription.
```

#### 🆕 `srt` with Speaker Labels
```
1
00:00:00,000 --> 00:00:02,500
[SPEAKER_00] Hello there.

2
00:00:02,500 --> 00:00:04,000
[SPEAKER_01] How are you today?
```

#### 🆕 `vtt` with Speaker Labels
```
WEBVTT

00:00:00.000 --> 00:00:02.500
[SPEAKER_00] Hello there.

00:00:02.500 --> 00:00:04.000
[SPEAKER_01] How are you today?
```

## WhisperX Features Guide

### Speaker Diarization

**What it does**: Identifies and labels different speakers in your audio files using advanced neural networks.

**Requirements**:
- Hugging Face token (free)
- ~2GB additional GPU memory
- Clear audio with distinguishable speakers

**Best Results**:
- 2-10 speakers
- Minimal background noise
- Clear speech separation
- Audio longer than 30 seconds

**Performance Impact**:
- Adds 20-50% to processing time
- GPU acceleration when available
- Memory usage increases by ~2GB

### Output Control

**What it does**: Control what content is included in the response.

**Options**:
- `text_only`: Only return the transcribed text
- `timestamps_only`: Only return timing information
- `both`: Return both text and timestamps (default)

### Troubleshooting

#### Diarization Issues
1. **"Pipeline not found" Error**:
   - Check HUGGINGFACE_TOKEN is set correctly
   - Verify you've accepted the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Ensure internet connection for model download

2. **Poor Diarization Results**:
   - Use `min_speakers`/`max_speakers` to constrain the search
   - Ensure clear audio with distinct speakers
   - Try shorter audio segments for better accuracy

3. **Memory Issues**:
   - Ensure sufficient GPU memory (>4GB free recommended)
   - Consider processing shorter audio segments

### Supported Audio Formats
- WAV
- MP3
- FLAC
- M4A
- And other formats supported by FFmpeg

## OpenAI API Compatibility

This API is fully compatible with OpenAI's Whisper API. You can:

1. **Use existing OpenAI client libraries** by pointing them to your server
2. **Drop-in replacement** for OpenAI Whisper API calls
3. **Same parameter names and response formats** as OpenAI
4. **Extended WhisperX features** that are ignored by standard OpenAI clients

### Migration from OpenAI
Simply change your base URL from `https://api.openai.com` to `http://localhost:3333` and optionally remove the API key requirement.

## Optimization & Performance Tuning

WhisperX handles many optimizations automatically, but you can still fine-tune performance for your specific needs. 

### Quick Start - Environment Variables:
```bash
# Customer feedback optimized (addresses accuracy issues)
export WHISPERX_COMPUTE_TYPE=float32
export WHISPERX_BATCH_SIZE=8
export WHISPERX_CHAR_ALIGN=true
export MAX_CONCURRENT_REQUESTS=4

# Balanced production (speed vs accuracy)
export WHISPERX_COMPUTE_TYPE=float16
export WHISPERX_BATCH_SIZE=12
export MAX_CONCURRENT_REQUESTS=5

# H100 high throughput (maximum concurrency)
export WHISPERX_COMPUTE_TYPE=float16
export WHISPERX_BATCH_SIZE=16
export MAX_CONCURRENT_REQUESTS=15
```

📋 **See [config_examples.env](config_examples.env) for complete configuration profiles**

**📖 For comprehensive optimization guidance, see [WHISPERX_OPTIMIZATION_GUIDE.md](WHISPERX_OPTIMIZATION_GUIDE.md)**

**🚀 For concurrent processing & GPU optimization, see [CONCURRENT_PROCESSING_GUIDE.md](CONCURRENT_PROCESSING_GUIDE.md)**

**🐳 For Docker deployment updates, see [DEPLOYMENT_UPDATES.md](DEPLOYMENT_UPDATES.md)**

### What's Different from faster-whisper:
- **Temperature, beam_size, condition_on_previous_text**: Now handled internally for better defaults
- **VAD parameters**: Automatic VAD preprocessing with pyannote/silero
- **Batch processing**: Optimized internal batching
- **Word alignment**: Phoneme-based alignment for better accuracy
- **Concurrent Processing**: Built-in support for multiple simultaneous transcriptions
- **GPU Optimization**: Memory-aware request management and monitoring

### Concurrent Processing Features:
- **RTX 3090**: Up to 4 concurrent requests (24GB VRAM) - **3600+ minutes/hour capacity**
- **H100**: Up to 15 concurrent requests (80GB VRAM) - **22,500+ minutes/hour capacity**
- **Memory Monitoring**: Real-time GPU memory tracking at `/processing-status`
- **Request Management**: Automatic concurrency control and cleanup
- **Queue Support**: Configurable request queuing for overflow handling

## Production Deployment

For production use, consider:
- Using a reverse proxy (nginx)
- Adding authentication
- Using HTTPS
- Rate limiting
- Error handling improvements
- Load balancing for multiple instances

## Performance Notes

- **GPU Recommended**: NVIDIA GPU with 4GB+ VRAM for optimal performance
- **CPU Fallback**: Will work on CPU but significantly slower
- **Memory Usage**: ~2-4GB GPU memory for transcription, +2GB for diarization
- **Processing Speed**: **15x faster than real-time** on RTX 3090 (measured)