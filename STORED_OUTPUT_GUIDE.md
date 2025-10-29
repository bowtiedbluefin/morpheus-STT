# Using `stored_output` Parameter

## Overview

The `stored_output` parameter controls **how your transcription is processed and returned**:

- **`stored_output=true`** → Async mode: Immediate response with `job_id`, retrieve results later
- **`stored_output=false`** → Sync mode: Wait for completion, results returned inline

This gives you flexibility in how you integrate the transcription API.

## When to Use Each Mode

### Use Async Mode (`stored_output=true`) When:
- Processing long audio files (>5 minutes)
- You don't want to keep HTTP connection open
- Building a queue-based system
- You want to poll for results at your convenience
- Processing multiple files concurrently

### Use Sync Mode (`stored_output=false`) When:
- Processing short audio clips (<2 minutes)
- You need immediate results
- Building a real-time transcription UI
- Simple integration without polling logic
- You want specific response formats (SRT, VTT, plain text)

## Async Mode (`stored_output=true`)

### How It Works

```
1. Submit Request → Get job_id immediately
2. Server processes in background
3. Poll GET /v1/jobs/{job_id} for status
4. Results available for 24 hours
```

### Example: Basic Async

```bash
# 1. Submit job
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "stored_output=true"
```

Response:
```json
{
  "job_id": "abc123-def456-...",
  "status": "in_progress",
  "message": "Transcription job started. Results will be stored locally...",
  "retrieval_info": {
    "method": "GET /v1/jobs/{job_id}",
    "retention_seconds": 86400,
    "note": "Poll this endpoint to check status and retrieve results"
  }
}
```

```bash
# 2. Check status (lightweight)
curl "http://localhost:3333/v1/jobs/abc123...?include_result=false"
```

Response (processing):
```json
{
  "job_id": "abc123...",
  "status": "processing",
  "progress": "Transcribing audio",
  "elapsed_time": 15.3
}
```

```bash
# 3. Get full results when complete
curl "http://localhost:3333/v1/jobs/abc123...?include_result=true"
```

Response (completed):
```json
{
  "job_id": "abc123...",
  "status": "completed",
  "progress": "Done",
  "elapsed_time": 45.2,
  "result": {
    "result": {
      "text": "Full transcription text...",
      "segments": [
        {
          "text": " Hello world",
          "start": 0.0,
          "end": 1.5,
          "speaker": "SPEAKER_00"
        }
      ]
    }
  }
}
```

### Example: Async with AWS Callback

You can **optionally** also upload results to your S3 bucket:

```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "stored_output=true" \
  -F "upload_presigned_url=https://your-bucket.s3.amazonaws.com/..."
```

Response:
```json
{
  "job_id": "abc123...",
  "status": "in_progress",
  "retrieval_info": {
    "method": "GET /v1/jobs/{job_id}",
    "note": "Poll this endpoint..."
  },
  "callback_info": {
    "upload_url": "https://your-bucket.s3.amazonaws.com/...",
    "note": "Results will also be uploaded to your S3 bucket"
  }
}
```

Results will be:
1. Stored locally (retrievable via GET endpoint)
2. Uploaded to your S3 bucket

### Python Example: Async Mode

```python
import requests
import time

# Submit job
response = requests.post(
    'http://localhost:3333/v1/audio/transcriptions',
    files={'file': open('audio.mp3', 'rb')},
    data={
        'enable_diarization': 'true',
        'stored_output': 'true'  # Async mode
    }
)
job_id = response.json()['job_id']
print(f"Job submitted: {job_id}")

# Poll for completion
while True:
    status = requests.get(
        f'http://localhost:3333/v1/jobs/{job_id}?include_result=false'
    ).json()
    
    print(f"Status: {status['status']} - {status.get('progress', '')}")
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print(f"Error: {status.get('error')}")
        exit(1)
    
    time.sleep(2)

# Get full results
result = requests.get(
    f'http://localhost:3333/v1/jobs/{job_id}?include_result=true'
).json()

transcription = result['result']['result']['text']
segments = result['result']['result']['segments']
print(f"Transcription: {transcription}")
```

## Sync Mode (`stored_output=false`)

### How It Works

```
1. Submit Request
2. Wait for processing
3. Get full results in response
4. No polling needed
```

### Example: Basic Sync

```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "stored_output=false"
```

Response (immediate, after processing):
```json
{
  "result": {
    "text": "Full transcription text...",
    "segments": [
      {
        "text": " Hello world",
        "start": 0.0,
        "end": 1.5,
        "speaker": "SPEAKER_00"
      }
    ],
    "language": "en"
  }
}
```

### Example: Different Response Formats

Sync mode supports multiple response formats via the `response_format` parameter:

**JSON (default):**
```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "stored_output=false" \
  -F "response_format=json"
```

**Plain Text:**
```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "stored_output=false" \
  -F "response_format=text"
```

Response:
```
Hello world. This is a transcription.
```

**SRT Subtitles:**
```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "stored_output=false" \
  -F "response_format=srt"
```

Response:
```
1
00:00:00,000 --> 00:00:01,500
Hello world

2
00:00:01,500 --> 00:00:03,000
This is a transcription
```

**VTT Subtitles:**
```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "stored_output=false" \
  -F "response_format=vtt"
```

### Example: Control Output Content

Use `output_content` to control what's included:

```bash
# Text only (no timestamps)
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "stored_output=false" \
  -F "output_content=text_only"

# Timestamps only (no separate text field)
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "stored_output=false" \
  -F "output_content=timestamps_only"

# Both (default)
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "stored_output=false" \
  -F "output_content=both"
```

### Python Example: Sync Mode

```python
import requests

# Submit job and wait for results
response = requests.post(
    'http://localhost:3333/v1/audio/transcriptions',
    files={'file': open('audio.mp3', 'rb')},
    data={
        'enable_diarization': 'true',
        'stored_output': 'false'  # Sync mode
    }
)

# Results available immediately
result = response.json()
transcription = result['result']['text']
segments = result['result']['segments']
print(f"Transcription: {transcription}")
```

## Comparison

| Feature | Async (`stored_output=true`) | Sync (`stored_output=false`) |
|---------|------------------------------|------------------------------|
| **Response Time** | Immediate (~50ms) | After processing (30s-5min) |
| **Response Content** | Job metadata | Full transcription |
| **Polling Required** | Yes | No |
| **Best For** | Long files, queues | Short files, real-time |
| **HTTP Connection** | Short-lived | Long-lived |
| **Result Storage** | 24 hours | Not stored |
| **Response Formats** | JSON only | JSON, text, SRT, VTT |
| **AWS Callback** | Optional | N/A |

## Migration from Old Behavior

### Old Way (deprecated)
```bash
# Async mode was triggered by upload_presigned_url
curl -F "file=@audio.mp3" \
     -F "upload_presigned_url=https://..."
```

### New Way (recommended)
```bash
# Async mode explicitly controlled by stored_output
curl -F "file=@audio.mp3" \
     -F "stored_output=true" \
     -F "upload_presigned_url=https://..."  # Optional callback
```

**Note**: The old way still works for backward compatibility, but we recommend using `stored_output` explicitly.

## Configuration

### Environment Variables

```bash
# How long to keep results (default: 24 hours)
JOB_RESULT_RETENTION_SECONDS=86400

# Where to store results locally
JOB_RESULTS_DIR=/tmp/whisper-job-results
```

### Disable Server-Side Storage

If you only want sync mode and don't want any local storage:

```bash
JOB_RESULT_RETENTION_SECONDS=0
```

## Best Practices

### For Async Mode

1. **Poll Efficiently**: Use `include_result=false` while polling to reduce bandwidth
2. **Handle Failures**: Check for `status: "failed"` and handle errors
3. **Set Timeouts**: Don't poll forever, set a max timeout
4. **Consider Retention**: Results expire after 24 hours by default

### For Sync Mode

1. **Timeout Handling**: Set appropriate HTTP timeouts (5+ minutes for long files)
2. **Connection Stability**: Ensure stable network connection
3. **File Size Limits**: Consider async mode for files >10 minutes
4. **Response Format**: Choose appropriate format for your use case

## Common Patterns

### Pattern 1: Simple Transcription

```python
# Just need transcription, don't care about storage
response = requests.post(url, files=files, data={'stored_output': 'false'})
print(response.json()['result']['text'])
```

### Pattern 2: Batch Processing

```python
# Submit multiple jobs
job_ids = []
for audio_file in audio_files:
    response = requests.post(url, files={'file': audio_file}, 
                           data={'stored_output': 'true'})
    job_ids.append(response.json()['job_id'])

# Poll all jobs
for job_id in job_ids:
    while True:
        status = requests.get(f'{url}/v1/jobs/{job_id}').json()
        if status['status'] == 'completed':
            process_result(status['result'])
            break
        time.sleep(5)
```

### Pattern 3: Webhook-Style with Fallback

```python
# Upload to S3 but also keep on server for retrieval
response = requests.post(url, files=files, data={
    'stored_output': 'true',
    'upload_presigned_url': s3_url  # Primary method
})
job_id = response.json()['job_id']

# If S3 upload fails, can still retrieve from server
result = requests.get(f'{url}/v1/jobs/{job_id}').json()
```

## Summary

- **Use `stored_output=true`** for async processing with local storage
- **Use `stored_output=false`** for sync processing with inline results
- **Add `upload_presigned_url`** optionally for AWS S3 callbacks
- Results stored for 24 hours (configurable)
- Both modes support all transcription features

Choose the mode that best fits your integration needs!

