# Timeout Configuration Guide

## Important: Two Different Timeouts

### 1. Transcription Processing Time ✅ NO TIMEOUT
**What it is:** The time it takes to transcribe and diarize the audio file.

**Current setting:** ✅ **No timeout** - runs as long as needed

**Why no timeout:** 
- Large audio files can take 10+ minutes to transcribe
- Processing time varies with file length, number of speakers, and concurrency
- Under max concurrency, jobs may queue before processing starts
- Transcription happens in background tasks, not blocking HTTP requests

**Example processing times:**
- 1-minute audio: ~5-10 seconds
- 10-minute audio: ~45-90 seconds  
- 60-minute audio: ~5-15 minutes
- With max concurrency, add queue wait time

### 2. Upload Timeout ⏱️ CONFIGURABLE
**What it is:** The time allowed to upload the transcription result JSON to your S3/presigned URL.

**Current setting:** 
- **Default:** 600 seconds (10 minutes)
- **Configurable via:** `UPLOAD_TIMEOUT_SECONDS` environment variable

**Why it matters:**
- Large audio files produce large result files (many segments, speaker data)
- A 60-minute audio file with 10 speakers might generate 5-10MB JSON
- Under slow network conditions, upload can take time
- Default 2 minutes was too short for large results

## Configuration

### Set Upload Timeout

**In .env file:**
```bash
# 10 minutes (default)
UPLOAD_TIMEOUT_SECONDS=600

# 20 minutes for very large files
UPLOAD_TIMEOUT_SECONDS=1200

# 30 minutes for extreme cases
UPLOAD_TIMEOUT_SECONDS=1800
```

**In environment:**
```bash
export UPLOAD_TIMEOUT_SECONDS=1200
./start_server.sh
```

**In Docker/Kubernetes:**
```yaml
env:
  - name: UPLOAD_TIMEOUT_SECONDS
    value: "1200"  # 20 minutes
```

### Recommended Settings

Based on your typical file sizes:

| Max Audio Length | Typical Result Size | Recommended Timeout |
|-----------------|-------------------|-------------------|
| < 10 minutes | < 1 MB | 300s (5 min) |
| 10-30 minutes | 1-3 MB | 600s (10 min) ✅ Default |
| 30-60 minutes | 3-10 MB | 1200s (20 min) |
| 60+ minutes | 10+ MB | 1800s (30 min) |

**Network considerations:**
- Fast network (100+ Mbps): Use defaults
- Slow network (10 Mbps): Double the timeout
- International upload: Add 50% buffer

### Check Current Settings

**Via health endpoint:**
```bash
curl http://localhost:8000/health | jq '.timeouts'
```

**Output:**
```json
{
  "upload_timeout_seconds": 600,
  "upload_timeout_minutes": 10.0,
  "note": "Timeout for uploading results to S3. Transcription itself has no timeout."
}
```

**In server logs on startup:**
```
🎯 Concurrent processing limit: 12 jobs
⏱️  Upload timeout: 600s (10.0 minutes)
```

## What Each Timeout Controls

### Upload Timeout Breakdown

The `UPLOAD_TIMEOUT_SECONDS` controls three sub-timeouts:

```python
timeout = httpx.Timeout(
    connect=30.0,                     # Fixed: 30s to establish connection
    read=UPLOAD_TIMEOUT_SECONDS,      # Your setting: time to read S3 response
    write=UPLOAD_TIMEOUT_SECONDS,     # Your setting: time to write data to S3
    pool=10.0                         # Fixed: 10s to get connection from pool
)
```

**Connect timeout (30s):** Fixed, usually sufficient. Only need to increase if S3 is extremely slow to respond.

**Read/Write timeout:** Your configurable setting. This is what you increase for large files.

**Pool timeout (10s):** Fixed, controls connection pooling. Rarely needs adjustment.

## Timeline Example

Here's what happens with a large file under max concurrency:

```
Job Lifecycle (60-minute audio file):

00:00 - Request received
        ├─ Return immediate response with job_id
        └─ Job enters queue

02:30 - GPU slot becomes available
        └─ Job starts processing

03:00 - Download audio from presigned URL (30s)
        └─ Progress: "Downloading audio file"

15:00 - Transcription complete (12 minutes)
        └─ Progress: "Transcribing audio"
        └─ Result: 8 MB JSON file

15:05 - Upload to S3 (5 seconds, fast network)
        └─ Progress: "Uploading results"
        ├─ Using UPLOAD_TIMEOUT_SECONDS = 600s
        └─ Actual upload time: 5s (well within timeout)

15:05 - Job complete
        └─ Status: "completed"

Total time: 15 minutes 5 seconds
  - Queue wait: 2m 30s
  - Download: 30s
  - Transcription: 12m (NO TIMEOUT)
  - Upload: 5s (timeout: 10m)
```

## Error Messages

### If Upload Times Out

**Error you'll see:**
```json
{
  "job_id": "abc-123",
  "status": "failed",
  "error": "Write timeout after 600s uploading to URL. Large file may need longer timeout.",
  "error_type": "HTTPException"
}
```

**Solution:**
1. Increase `UPLOAD_TIMEOUT_SECONDS` in your environment
2. Restart the server
3. Retry the request

**Check result file size:**
```bash
# In logs, look for:
📤 Uploading result (8450123 bytes) to presigned URL...
#                    ^^^^^^^^
# Size in bytes
```

If result is > 5MB and network is slow, increase timeout.

## Retry Logic

The system automatically retries uploads on timeout:

```
Attempt 1: Upload fails after 600s
⏳ Wait 2 seconds
Attempt 2: Upload fails after 600s  
⏳ Wait 4 seconds
Attempt 3: Upload fails after 600s
❌ Give up, report failure
```

**Total time for 3 attempts:** `(600s × 3) + 2s + 4s = 1806s (~30 minutes)`

If all 3 attempts time out, you definitely need to increase the timeout.

## Best Practices

### 1. Start Conservative
```bash
UPLOAD_TIMEOUT_SECONDS=1200  # 20 minutes
```
Better to have extra buffer than timeout failures.

### 2. Monitor Actual Upload Times
Check logs for actual upload duration:
```
📤 Uploading result (8450123 bytes) to presigned URL... (attempt 1/3)
✅ Successfully uploaded result to user-provided URL (8450123 bytes)
```

Time between these lines = actual upload time.

### 3. Adjust Based on Data
If uploads consistently complete in 30 seconds, you can reduce timeout to 300s (5 min).

If uploads occasionally take 8 minutes, set timeout to 1200s (20 min) for buffer.

### 4. Consider Network Conditions
- Server to S3 in same region: Low timeout OK
- Server to S3 cross-region: Increase by 50%
- Server to customer S3 (unknown location): Use conservative timeout

## Testing Upload Timeout

### Test with Large File
```bash
# Submit large audio file
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@large_60min_audio.mp3" \
  -F "upload_presigned_url=$PRESIGNED_URL" \
  -F "enable_diarization=true"

# Monitor job
JOB_ID="returned-job-id"
watch -n 5 "curl -s http://localhost:8000/v1/jobs/$JOB_ID | jq"
```

**Watch for:**
- Progress: "Uploading results" - this is when timeout matters
- If it stays here for > 5 minutes on fast network, increase timeout
- Check server logs for upload size and duration

### Simulate Slow Network
```bash
# Use a presigned URL with network throttling
# Or monitor upload time in production under real conditions
```

## Summary

| Setting | What It Controls | Current Value | Configurable? |
|---------|-----------------|---------------|---------------|
| Transcription time | How long transcription can run | ∞ (no limit) | ❌ No (intentionally unlimited) |
| Upload timeout | How long upload to S3 can take | 600s (10 min) | ✅ Yes - `UPLOAD_TIMEOUT_SECONDS` |
| Connect timeout | Time to establish connection | 30s | ❌ No (usually sufficient) |
| Pool timeout | Time to get connection from pool | 10s | ❌ No (usually sufficient) |

**Key Point:** Transcription can take as long as needed. Only the final upload has a timeout, which you can now configure.

## FAQ

**Q: My transcription takes 15 minutes. Will it timeout?**
A: No. Transcription has no timeout. It runs as long as needed.

**Q: I got a timeout error. What should I increase?**
A: Check the error message. If it says "Write timeout" or "Read timeout", increase `UPLOAD_TIMEOUT_SECONDS`.

**Q: What if I set the timeout too high?**
A: No problem. It's just a maximum allowed time. Fast uploads still complete quickly. You'll just wait longer before a genuine failure is detected.

**Q: Can I set different timeouts for different requests?**
A: No, it's server-wide. Set it to accommodate your largest files.

**Q: Does this timeout affect sync mode?**
A: Sync mode doesn't upload to presigned URLs, so this timeout doesn't apply. Sync mode just waits for transcription to complete (unlimited) and returns the result in the API response.

**Q: What about the presigned URL expiration?**
A: That's separate. Your presigned URL must be valid long enough to cover:
- Queue wait time
- Transcription time  
- Upload time
- Retry attempts if needed

Recommend presigned URLs valid for at least 2 hours for large files.

