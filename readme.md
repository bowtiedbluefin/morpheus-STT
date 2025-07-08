# Whisper Transcription API

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's Whisper API
- **Multiple Output Formats**: JSON, text, SRT, VTT with flexible content options
- **Advanced Processing**: Batched processing for long audio files
- **Word-Level Timestamps**: Precise word-level timing information
- **🆕 Speaker Diarization**: Identify and label different speakers in audio
- **🆕 Profanity Filtering**: Automatically censor profane language
- **GPU Acceleration**: CUDA support for faster processing
- **Multiple Languages**: Auto-detection and manual language specification

## Installation

### Install dependencies
```bash
pip install -r requirements.txt
```

### Optional: Enhanced Features Setup

#### Speaker Diarization Setup
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

### Basic usage
```python
from faster_whisper import WhisperModel

# Run on GPU with FP16
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# For long files use chunking
segments, info = model.transcribe("audio.wav", beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500), initial_prompt="Transcribe audio")

# Process and save the transcript
with open("transcript.txt", "w") as f:
    for segment in segments:
        f.write(f"{segment.text}\n")
```

## FastAPI Server Setup

### Prerequisites
1. Python 3.8+ installed
2. NVIDIA GPU with CUDA support (optional but recommended)
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
python3 python_server.py
```

The server will start on `http://localhost:3333`

### Model Download

**No manual model download required!** The Whisper model (large-v3, ~3GB) will be automatically downloaded on first run and saved to the `./models/` directory. The download may take a few minutes depending on your internet connection.

- **First run**: Model downloads automatically, may take 5-10 minutes
- **Subsequent runs**: Uses cached model, starts immediately
- **Storage**: Models are stored in `./models/` (excluded from git via `.gitignore`)

## API Usage

### Endpoints

- **GET** `/` - Welcome message and basic info
- **GET** `/transcribe/` - Information about the transcription endpoint
- **POST** `/v1/audio/transcriptions` - OpenAI-compatible transcription endpoint (recommended)
- **POST** `/transcribe/` - Alias to the main endpoint (same functionality)
- **GET** `/docs` - Interactive API documentation (Swagger UI)

Both POST endpoints support identical functionality with full OpenAI API compatibility plus extended features.

### Using the API

#### 1. Web Interface (Easiest)
Visit `http://localhost:3333/docs` in your browser to use the interactive API documentation where you can upload files directly and test all parameters.

#### 2. cURL Command Examples

##### Basic Usage (OpenAI Compatible)
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio/file.wav" \
  -F "model=whisper-1" \
  --progress-bar
```

##### 🆕 Speaker Diarization Examples
```bash
# Basic diarization
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "enable_diarization=true" \
  -F "response_format=json"

# Diarization with speaker constraints
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  -F "response_format=json"

# SRT format with speaker labels
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "enable_diarization=true" \
  -F "response_format=srt" \
  -o subtitles_with_speakers.srt
```

##### 🆕 Profanity Filtering Examples
```bash
# Enable profanity filter
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "enable_profanity_filter=true" \
  -F "response_format=json"

# Combined features: diarization + profanity filter
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "enable_diarization=true" \
  -F "enable_profanity_filter=true" \
  -F "response_format=json"
```

##### All Available Options Example
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio/file.wav" \
  -F "model=whisper-1" \
  -F "language=en" \
  -F "prompt=This is a high-quality business meeting recording with clear audio." \
  -F "response_format=verbose_json" \
  -F "temperature=0.1" \
  -F "timestamp_granularities[]=word" \
  -F "output_content=both" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=5" \
  -F "enable_profanity_filter=true" \
  --progress-bar \
  --connect-timeout 30 \
  --max-time 3600 \
  -o transcription_with_words.json
```

##### OpenAI Standard Format Examples
```bash
# Basic OpenAI-style request
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"

# With word-level timestamps (OpenAI format)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"

# With segment-level timestamps (OpenAI format)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=segment"
```

##### Output Content Options (Extended Feature)
```bash
# Text only (no timestamps)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json" \
  -F "output_content=text_only" \
  -o text_only.json

# Timestamps only (no text)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=segment" \
  -F "output_content=timestamps_only" \
  -o timestamps_only.json

# Both text and timestamps (default)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=word" \
  -F "output_content=both" \
  -o complete_transcription.json
```

##### Language-Specific Transcription
```bash
# Spanish audio
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@spanish_audio.wav" \
  -F "model=whisper-1" \
  -F "language=es" \
  -F "prompt=Transcripción de una reunión de negocios en español."

# French audio
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@french_audio.wav" \
  -F "model=whisper-1" \
  -F "language=fr" \
  -F "prompt=Transcription d'un enregistrement audio en français."

# Auto-detect language (default)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@multilingual_audio.wav" \
  -F "model=whisper-1" \
  -F "prompt=Mixed language audio recording."
```

##### Different Response Formats
```bash
# JSON with segment timestamps (default)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=segment" \
  -o transcription_segments.json

# JSON with word-level timestamps
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json" \
  -F "timestamp_granularities[]=word" \
  -o transcription_words.json

# Verbose JSON with metadata (OpenAI format)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -o transcription_verbose.json

# Plain text output
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=text" \
  -o transcription.txt

# SRT subtitle format
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=srt" \
  -o subtitles.srt

# WebVTT subtitle format
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=vtt" \
  -o subtitles.vtt
```

##### Temperature Settings for Different Use Cases
```bash
# Deterministic output (most consistent)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "temperature=0.0"

# Slightly more creative (good for unclear audio)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "temperature=0.2"

# More creative output (use with caution)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "temperature=0.5"
```

##### Using the Alias Endpoint
```bash
# Both endpoints support identical functionality
curl -X POST "http://localhost:3333/transcribe/" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"
```

##### Advanced cURL Options
```bash
# With timeout and retry settings
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@large_audio.wav" \
  -F "model=whisper-1" \
  -F "language=en" \
  --connect-timeout 30 \
  --max-time 7200 \
  --retry 3 \
  --retry-delay 5 \
  --progress-bar

# Silent processing (no progress output)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  --silent \
  -o result.json

# Verbose debugging output
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  --verbose \
  -o debug_result.json

# Save response headers
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -D response_headers.txt \
  -o transcription.json

# Append to existing file
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=text" \
  >> combined_transcriptions.txt
```

#### 3. Python requests (OpenAI Compatible)
```python
import requests

url = "http://localhost:3333/v1/audio/transcriptions"
files = {"file": open("audio.wav", "rb")}
data = {
    "model": "whisper-1",
    "language": "en",
    "prompt": "This is a business meeting recording.",
    "response_format": "verbose_json",
    "temperature": 0.1,
    "timestamp_granularities[]": "word",
    # New features
    "enable_diarization": True,
    "min_speakers": 2,
    "max_speakers": 4,
    "enable_profanity_filter": True
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
form.append('model', 'whisper-1');
form.append('language', 'en');
form.append('prompt', 'Business meeting recording');
form.append('response_format', 'verbose_json');
form.append('temperature', '0.1');
form.append('timestamp_granularities[]', 'segment');
// New features
form.append('enable_diarization', 'true');
form.append('min_speakers', '2');
form.append('max_speakers', '4');
form.append('enable_profanity_filter', 'true');

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
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    print(transcript.text)
```

## API Parameters

### Required Parameters
- **`file`**: Audio file to transcribe (mp3, wav, m4a, flac, etc.)

### OpenAI Compatible Parameters
- **`model`**: Model to use (e.g., "whisper-1") - handled by API gateway
- **`language`**: Language code (ISO-639-1) like "en", "es", "fr" - improves accuracy
- **`prompt`**: Initial prompt to condition the model
- **`response_format`**: Output format - `json`, `text`, `srt`, `vtt`, `verbose_json`
- **`temperature`**: Sampling temperature (0.0-1.0) - 0.0 for deterministic output
- **`timestamp_granularities[]`**: Array or single value - `segment`, `word`

### Extended Parameters (Not in OpenAI API)
- **`output_content`**: Fine-grained control - `text_only`, `timestamps_only`, `both`
- **🆕 `enable_diarization`**: Enable speaker diarization (boolean, default: false)
- **🆕 `min_speakers`**: Minimum number of speakers (1-20, optional)
- **🆕 `max_speakers`**: Maximum number of speakers (1-20, optional)
- **🆕 `enable_profanity_filter`**: Enable profanity filtering (boolean, default: false)

### Response Formats

#### `json` (Simple JSON)
```json
{
  "text": "Hello, this is a transcription.",
  "segments": [...]  // or "words": [...] depending on granularity
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
      "speaker": "SPEAKER_00"
    },
    {
      "start": 2.5,
      "end": 4.0,
      "text": "How are you today?",
      "speaker": "SPEAKER_01"
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
  "segments": [...]  // or "words": [...]
}
```

#### 🆕 `verbose_json` with Enhanced Features
```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 44.08,
  "speakers": 2,
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
  ]
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

## 🆕 New Features Guide

### Speaker Diarization

**What it does**: Identifies and labels different speakers in your audio files.

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

### Profanity Filtering

**What it does**: Automatically censors profane language in transcriptions.

**Features**:
- Replaces profane words with asterisks
- Works with all output formats
- Minimal performance impact
- English language focus

**Customization**: The underlying library supports custom word lists for specific use cases.

### Troubleshooting New Features

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
   - Ensure sufficient GPU memory (>2GB free)
   - Consider processing shorter audio segments

#### Profanity Filter Issues
- Filter uses a standard English word list
- May not catch all variations or slang
- Consider the context of your use case

### External Access

#### Local Network Access
To allow access from other devices on your local network, the server is already configured to bind to `0.0.0.0:3333`. Other devices can access it using your computer's IP address:
```
http://YOUR_IP_ADDRESS:3333/v1/audio/transcriptions
```

#### Internet Access (Advanced)
For internet access, you'll need to:
1. Configure port forwarding on your router (port 3333)
2. Use your public IP address
3. Consider security implications and authentication

#### Production Deployment
For production use, consider:
- Using a reverse proxy (nginx)
- Adding authentication
- Using HTTPS
- Rate limiting
- Error handling improvements

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
4. **Extended features** that are ignored by standard OpenAI clients

### Migration from OpenAI
Simply change your base URL from `https://api.openai.com` to `http://localhost:3333` and optionally remove the API key requirement.

## Advanced cURL Options & Configuration

### Basic cURL Options

#### Connection & Transfer Options
```bash
# Basic request with timeout
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  --connect-timeout 30 \
  --max-time 3600 \
  -F "file=@audio.wav" \
  -F "model=whisper-1"

# Show detailed progress
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  --progress-bar \
  -F "file=@audio.wav" \
  -F "model=whisper-1"

# Silent mode (no progress)
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  --silent \
  -F "file=@audio.wav" \
  -F "model=whisper-1"

# Verbose output for debugging
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  --verbose \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

#### Output Options
```bash
# Save to specific file
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -o transcription.json
```

#### Network Options
```bash
# Use specific network interface
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  --interface eth0 \
  -F "file=@audio.wav" \
  -F "model=whisper-1"

# Use proxy
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  --proxy http://proxy.example.com:8080 \
  -F "file=@audio.wav" \
  -F "model=whisper-1"

# Follow redirects
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  --location \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

#### Transcription Parameters
```python
segments, info = model.transcribe(
    audio_file,
    
    # Language detection
    language=None,              # Auto-detect (default) or specify: "en", "es", "fr", etc.
    
    # Beam search parameters
    beam_size=5,                # Number of beams (1-10, higher = more accurate but slower)
    best_of=5,                  # Number of candidates to consider
    patience=1.0,               # Beam search patience factor
    
    # Voice Activity Detection (VAD)
    vad_filter=True,            # Enable VAD filtering
    vad_parameters=dict(
        threshold=0.5,          # VAD threshold (0.0-1.0)
        min_speech_duration_ms=250,     # Minimum speech duration
        max_speech_duration_s=float("inf"),  # Maximum speech duration
        min_silence_duration_ms=2000,   # Minimum silence duration
        window_size_samples=1024,       # VAD window size
        speech_pad_ms=400,              # Padding around speech
    ),
    
    # Temperature and sampling
    temperature=0.0,            # Sampling temperature (0.0 = deterministic)
    compression_ratio_threshold=2.4,  # Compression ratio threshold
    log_prob_threshold=-1.0,    # Log probability threshold
    no_speech_threshold=0.6,    # No speech threshold
    
    # Conditioning
    initial_prompt=None,        # Initial prompt to condition the model
    prefix=None,                # Prefix for the transcription
    suppress_blank=True,        # Suppress blank outputs
    suppress_tokens=[-1],       # Token IDs to suppress
    
    # Output options
    without_timestamps=False,   # Disable timestamp prediction
    max_initial_timestamp=1.0,  # Maximum initial timestamp
    word_timestamps=False,      # Enable word-level timestamps
    prepend_punctuations="\"'"¿([{-",     # Punctuations to prepend
    append_punctuations="\"'.。,，!！?？:：")]}、",  # Punctuations to append
    
    # Decoding options
    repetition_penalty=1.0,     # Repetition penalty
    no_repeat_ngram_size=0,     # No repeat n-gram size
    prompt_reset_on_temperature=0.5,  # Reset prompt on temperature
)
```

### Example Advanced Configurations

#### High Accuracy Configuration
```python
# In python_server.py, modify the transcribe call:
segments, info = model.transcribe(
    temp_file.name,
    beam_size=10,               # Higher beam size
    best_of=10,                 # More candidates
    temperature=0.0,            # Deterministic
    vad_filter=True,
    vad_parameters=dict(
        threshold=0.3,          # Lower threshold for more sensitivity
        min_silence_duration_ms=1000,
    ),
    initial_prompt="This is a high-quality audio recording.",
)
```

#### Language-Specific Configuration
```python
# For specific language (e.g., Spanish)
segments, info = model.transcribe(
    temp_file.name,
    language="es",              # Spanish
    beam_size=5,
    initial_prompt="Transcripción en español.",
)
```