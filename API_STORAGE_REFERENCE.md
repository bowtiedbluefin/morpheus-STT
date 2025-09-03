# 📡 morpheus-STT Storage API Reference

## Quick Reference for New Storage Features

### 🔗 Endpoints

#### 1. Health Check
```
GET /health
```
**New fields in response:**
```json
{
  "storage_clients": {
    "r2_enabled": true,
    "s3_enabled": true
  }
}
```

#### 2. R2 Upload (NEW)
```
POST /v1/uploads
```
**Parameters:**
- `file` (required): Audio file to upload
- `bucket` (required): R2 bucket name  
- `key` (optional): Custom object key (auto-generated if not provided)

**Response:**
```json
{
  "status": "success",
  "r2_bucket": "sonartext-uploads",
  "r2_key": "uploads/20241201_143022_123456.wav",
  "storage_url": "s3://sonartext-uploads/uploads/20241201_143022_123456.wav"
}
```

#### 3. Enhanced Transcription (UPDATED)
```
POST /v1/audio/transcriptions
```

## 🔤 New Parameters Reference

### Input Sources (choose ONE)

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `file` | File | No* | Direct file upload | `@audio.wav` |
| `r2_bucket` + `r2_key` | str + str | No* | R2 object reference | `sonartext-uploads` + `uploads/file.wav` |
| `s3_presigned_url` | str | No* | Pre-signed S3 URL | `https://bucket.s3.amazonaws.com/file?...` |

*One input source is required

### Output Destination (optional)

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `output_bucket` | str | No | Target bucket for results | `transcription-results` |
| `output_key` | str | No | Object key (auto-generated if not provided) | `results/meeting_001.json` |
| `output_use_r2` | bool | No | Use R2 (true) or S3 (false) | `true` |

### All Existing Parameters Unchanged

- `enable_diarization`, `response_format`, `language`, `model`
- `prompt`, `temperature`, `timestamp_granularities`, `output_content`
- `clustering_threshold`, `segmentation_threshold`, `min_speaker_duration`
- `speaker_confidence_threshold`, `speaker_smoothing_enabled`, etc.

## 🔧 Environment Variables

### Required for R2
```bash
R2_ENDPOINT=https://your_account_id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
```

### Optional for S3 Result Export (NOT needed for pre-signed URLs)
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
```

## 📝 cURL Examples

### R2 Workflow
```bash
# Upload to R2
curl -X POST localhost:3333/v1/uploads \
  -F 'file=@audio.wav' \
  -F 'bucket=uploads'

# Transcribe from R2 with result export
curl -X POST localhost:3333/v1/audio/transcriptions \
  -d 'r2_bucket=uploads' \
  -d 'r2_key=uploads/20241201_143022_123456.wav' \
  -d 'output_bucket=results' \
  -d 'output_use_r2=true' \
  -d 'response_format=verbose_json'
```

### S3 Pre-signed URL
```bash
curl -X POST localhost:3333/v1/audio/transcriptions \
  -d 's3_presigned_url=https://bucket.s3.amazonaws.com/file?...' \
  -d 'output_bucket=enterprise-results' \
  -d 'response_format=json'
```

### Backward Compatibility  
```bash
# Still works exactly as before
curl -X POST localhost:3333/v1/audio/transcriptions \
  -F 'file=@audio.wav' \
  -F 'response_format=verbose_json'
```

## ✅ Response Changes

### New Storage Info (when output_bucket is used)
```json
{
  "status": "success",
  "text": "Transcribed text...",
  "result": { /* transcription results */ },
  "storage_info": {
    "uploaded": true,
    "storage_url": "s3://results-bucket/transcriptions/20241201_result.json",
    "bucket": "results-bucket",
    "key": "transcriptions/20241201_result.json",
    "service": "R2"
  }
}
```

## 🚨 Error Codes

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| 400 | "One input source required" | No input provided | Provide file, R2 reference, or S3 URL |
| 400 | "Only one input source allowed" | Multiple inputs | Use only one input method |
| 404 | "File not found in R2" | Invalid R2 key | Check bucket and key exist |
| 500 | "R2 client not configured" | Missing R2 credentials | Set R2 environment variables |
| 500 | "Failed to download from pre-signed URL" | Invalid/expired URL | Generate new URL |

---

**Backward Compatibility**: ✅ All existing API calls work unchanged 