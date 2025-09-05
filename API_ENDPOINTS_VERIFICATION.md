# 📡 API Endpoints Verification Guide

## Available Endpoints

### **1. Health Check**
```
GET /health
```
**Purpose**: Server status and feature availability  
**Authentication**: None required  
**Response**: JSON with server status

**Test:**
```bash
curl https://your-server/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "device": "cuda",
  "models_loaded": {
    "transcription": true,
    "alignment": true,
    "diarization": true
  },
  "storage_clients": {
    "r2_enabled": true,
    "s3_enabled": false
  },
  "features": [...]
}
```

---

### **2. R2 Direct Upload (NEW)**
```
POST /v1/uploads
```
**Purpose**: Upload files directly to R2 bucket  
**Content-Type**: `multipart/form-data`  
**Authentication**: R2 credentials in server environment

**Parameters:**
- `file` (required): Audio file to upload
- `bucket` (required): R2 bucket name  
- `key` (optional): Custom object key

**Test:**
```bash
curl -X POST https://your-server/v1/uploads \
  -F 'file=@test_audio.wav'
```

**Expected Response:**
```json
{
  "status": "success",
  "bucket": "default-uploads",
  "key": "uploads/20241201_143022_123456.wav",
  "file_size": 1024000,
  "storage_url": "s3://default-uploads/uploads/20241201_143022_123456.wav"
}
```

---

### **3. Enhanced Transcription (UPDATED)**
```
POST /v1/audio/transcriptions
```
**Purpose**: Transcribe audio with multiple input sources and result export  
**Content-Type**: `multipart/form-data` OR `application/x-www-form-urlencoded`  
**Authentication**: Varies by input source

#### **Input Source 1: Direct Upload**
```bash
curl -X POST https://your-server/v1/audio/transcriptions \
  -F 'file=@audio.wav' \
  -F 'response_format=verbose_json'
```

#### **Input Source 2: R2 Object**
```bash
curl -X POST https://your-server/v1/audio/transcriptions \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'storage_key=uploads/20241201_143022_123456.wav' \
  -d 'response_format=verbose_json'
```

#### **Input Source 3: Pre-signed S3 URL**
```bash
curl -X POST https://your-server/v1/audio/transcriptions \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 's3_presigned_url=https://bucket.s3.amazonaws.com/file?X-Amz-Algorithm=...' \
  -d 'response_format=json'
```

#### **With Result Export**
```bash
curl -X POST https://your-server/v1/audio/transcriptions \
  -F 'file=@audio.wav' \
  -F 'output_bucket=results-bucket' \
  -F 'output_use_r2=true' \
  -F 'response_format=verbose_json'
```

**Expected Response (with result export):**
```json
{
  "status": "success",
  "text": "Transcribed text here...",
  "result": {
    "segments": [...],
    "language": "en"
  },
  "storage_info": {
    "uploaded": true,
    "storage_url": "s3://results-bucket/transcriptions/20241201_result.json",
    "bucket": "results-bucket", 
    "key": "transcriptions/20241201_result.json",
    "service": "R2"
  }
}
```

## Input Parameter Validation

### **Mutually Exclusive Inputs (choose ONE)**
- `file` (multipart file upload)
- `storage_key` (R2 object reference)
- `s3_presigned_url` (pre-signed S3 URL)

**Error if none provided:**
```json
{
  "detail": "One input source required: file upload, storage_key, or s3_presigned_url"
}
```

**Error if multiple provided:**
```json
{
  "detail": "Only one input source allowed at a time"
}
```

## Output Parameter Validation

### **Result Export (optional)**
- `output_bucket` (required if using export)
- `output_key` (optional - auto-generated if not provided)
- `output_use_r2` (boolean - default: false)

**Auto-generated key format:**
```
transcriptions/{timestamp}_result.json
Example: transcriptions/20241201_143022_123456_result.json
```

**Error if key without bucket:**
```json
{
  "detail": "output_bucket required when output_key is provided"
}
```

## Endpoint Security

### **R2 Upload Endpoint**
- Requires R2 credentials on server
- File access controlled by R2 bucket permissions
- No user authentication required (controlled at infrastructure level)

### **Pre-signed S3 URLs**
- No server-side AWS credentials required
- Security handled by the pre-signed URL itself
- URLs expire based on the signature (set by URL generator)

### **Result Export**
- Requires appropriate credentials for target storage
- R2 export: Uses server's R2 credentials
- S3 export: Requires AWS credentials on server 