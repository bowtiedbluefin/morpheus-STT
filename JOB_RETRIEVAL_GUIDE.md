# Job Result Retrieval Guide

## Overview

Your WhisperX server now supports storing transcription results locally and retrieving them via the GET endpoint. This allows customers to:

1. Submit a transcription job
2. Get immediate response with `job_id`
3. Poll or query the job status
4. **Retrieve full transcription results** when complete

## How It Works

### Architecture

```
Client Request → Server Processing → Local Storage + Optional AWS Upload
                                           ↓
                                    Retrievable for 24 hours
                                           ↓
                                    Automatic Cleanup
```

### Storage

- **Location**: Results stored in `/tmp/whisper-job-results/` (configurable)
- **Format**: JSON files named `{job_id}.json`
- **Retention**: 24 hours by default (configurable)
- **Cleanup**: Automatic hourly cleanup of expired results

## API Usage

### 1. Submit Transcription Job

**Async Mode with Local Storage**:
```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "language=en" \
  -F "stored_output=true"
```

Response:
```json
{
  "job_id": "abc123-def456-...",
  "status": "in_progress",
  "message": "Transcription job started..."
}
```

**Async Mode with Local Storage + AWS Callback**:
```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "language=en" \
  -F "stored_output=true" \
  -F "upload_presigned_url=https://your-bucket.s3.amazonaws.com/..."
```

Results will be uploaded to AWS **AND** stored locally for retrieval.

**Sync Mode (Inline Results)**:
```bash
curl -X POST http://localhost:3333/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "language=en" \
  -F "stored_output=false"
```

Waits for completion and returns full results immediately (no job_id, no polling needed).

### 2. Get Job Status (Lightweight)

```bash
curl "http://localhost:3333/v1/jobs/{job_id}?include_result=false"
```

Response:
```json
{
  "job_id": "abc123-def456-...",
  "status": "processing|completed|failed",
  "progress": "Transcribing audio",
  "start_time": 1234567890.123,
  "elapsed_time": 45.2,
  "result_available": true,
  "result_expires_at": 1234653890.123
}
```

### 3. Retrieve Full Results

```bash
curl "http://localhost:3333/v1/jobs/{job_id}?include_result=true"
```

Response:
```json
{
  "job_id": "abc123-def456-...",
  "status": "completed",
  "progress": "Done",
  "start_time": 1234567890.123,
  "elapsed_time": 120.5,
  "end_time": 1234568010.623,
  "result_available": true,
  "result": {
    "job_id": "abc123-def456-...",
    "status": "completed",
    "processing_time": 120.5,
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
      "language": "en",
      "speakers": ["SPEAKER_00", "SPEAKER_01"]
    }
  }
}
```

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# How long to keep job results (in seconds)
# Default: 86400 (24 hours)
JOB_RESULT_RETENTION_SECONDS=86400

# Where to store job results locally
# Default: /tmp/whisper-job-results
JOB_RESULTS_DIR=/tmp/whisper-job-results
```

### Disable Server-Side Storage

If you only want to use AWS callbacks without local storage:

```bash
JOB_RESULT_RETENTION_SECONDS=0
```

When disabled:
- No local storage
- GET endpoint returns only job metadata (no results)
- Results only sent to `upload_presigned_url`

## Workflow Examples

### Example 1: Simple Polling

```python
import requests
import time

# 1. Submit job
response = requests.post(
    'http://localhost:3333/v1/audio/transcriptions',
    files={'file': open('audio.mp3', 'rb')},
    data={
        'enable_diarization': 'true',
        'stored_output': 'true'  # Enable async mode
    }
)
job_id = response.json()['job_id']

# 2. Poll for completion
while True:
    status = requests.get(
        f'http://localhost:3333/v1/jobs/{job_id}?include_result=false'
    ).json()
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        raise Exception(status.get('error'))
    
    time.sleep(2)

# 3. Get full results
result = requests.get(
    f'http://localhost:3333/v1/jobs/{job_id}?include_result=true'
).json()

print(result['result']['result']['text'])
```

### Example 2: Webhook-Style Callback

```python
# Customer's workflow:
# 1. Submit job with callback URL
response = requests.post(
    'http://localhost:3333/v1/audio/transcriptions',
    files={'file': open('audio.mp3', 'rb')},
    data={
        'enable_diarization': 'true',
        'stored_output': 'true',  # Enable async mode
        'upload_presigned_url': 'https://my-bucket.s3.amazonaws.com/...'
    }
)
job_id = response.json()['job_id']

# 2. Results will be uploaded to S3 automatically
# 3. BUT customer can also retrieve via API if needed:
result = requests.get(f'http://localhost:3333/v1/jobs/{job_id}?include_result=true').json()
```

## Cleanup

### Automatic Cleanup

- Runs every hour automatically
- Removes result files older than `JOB_RESULT_RETENTION_SECONDS`
- Also runs after each job completion
- Logs cleanup activity

### Manual Cleanup

If needed, you can manually clean up the directory:

```bash
# Remove all results older than 24 hours
find /tmp/whisper-job-results -name "*.json" -mtime +1 -delete
```

## Monitoring

### Check Storage Usage

```bash
# Check how many results are stored
ls -lh /tmp/whisper-job-results/ | wc -l

# Check total size
du -sh /tmp/whisper-job-results/
```

### View Job Statistics

```bash
# Get all active jobs
curl http://localhost:3333/v1/jobs
```

## Benefits

### For Your Customers

1. **No AWS Required**: Can use the service without setting up S3
2. **Simple Integration**: Just poll the endpoint, no webhook setup
3. **Flexible**: Can use AWS callback OR polling OR both
4. **Reliable**: Results stored even if callback fails

### For You

1. **Simple Implementation**: No external dependencies (R2/S3) needed
2. **Low Cost**: Local filesystem storage
3. **Automatic Cleanup**: No manual intervention required
4. **Backward Compatible**: Existing customers with AWS callbacks still work

## Testing

Run the included test script:

```bash
python test_job_retrieval.py
```

This will:
1. Submit a test transcription
2. Poll for completion
3. Retrieve and display results
4. Save results to a file

## Troubleshooting

### Results Not Available

**Issue**: GET endpoint returns `result_note: "Result not available"`

**Causes**:
1. Result file was cleaned up (older than retention period)
2. Storage disabled (`JOB_RESULT_RETENTION_SECONDS=0`)
3. Job failed before results could be stored
4. Filesystem permission issues

**Solutions**:
- Increase `JOB_RESULT_RETENTION_SECONDS`
- Check filesystem permissions on `JOB_RESULTS_DIR`
- Verify disk space available

### Storage Growing Too Large

**Issue**: `/tmp/whisper-job-results` directory too large

**Solutions**:
- Decrease `JOB_RESULT_RETENTION_SECONDS`
- Move storage to larger partition via `JOB_RESULTS_DIR`
- Set up external monitoring/alerting

### Permission Errors

**Issue**: `Failed to save job result locally: Permission denied`

**Solution**:
```bash
# Create directory with proper permissions
sudo mkdir -p /tmp/whisper-job-results
sudo chown $(whoami):$(whoami) /tmp/whisper-job-results
chmod 755 /tmp/whisper-job-results
```

## Migration

### From Callback-Only to Hybrid

No changes needed! Just update to the new version and:

1. Results automatically stored locally
2. Existing callback URLs still work
3. New customers can use polling instead

### Disabling Local Storage

If you prefer callback-only:

```bash
# In .env
JOB_RESULT_RETENTION_SECONDS=0
```

## Summary

| Feature | Old Behavior | New Behavior |
|---------|-------------|--------------|
| AWS Callback | Required for async jobs | Optional |
| Result Storage | Only in customer's S3 | Local + Optional S3 |
| Result Retrieval | Not possible | GET /v1/jobs/{job_id} |
| Cleanup | Manual | Automatic |
| Retention | Indefinite (customer's S3) | Configurable (24h default) |

The new system is fully backward compatible while adding powerful new capabilities for customers who don't want to deal with AWS infrastructure.

