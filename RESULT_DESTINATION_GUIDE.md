# 📤 Result Destination Guide

## Where Users Specify Result Destinations

Users control where transcription results go through **API parameters** in their request to `/v1/audio/transcriptions`.

## **Option 1: Inline Results (Default)**

**When to use**: Real-time applications, small results, immediate processing

**How**: Send **no output parameters**
```bash
curl -X POST https://your-server/v1/audio/transcriptions \
  -F 'file=@audio.wav' \
  -F 'response_format=verbose_json'
  # No output_bucket = results returned inline
```

**Result**: Transcription comes back directly in HTTP response
```json
{
  "status": "success", 
  "text": "Transcribed text...",
  "result": {
    "segments": [...]
  }
}
```

## **Option 2: Cloud Storage Export**

**When to use**: Workflow automation, large results, archival, downstream processing

**How**: Add **output parameters** to the same API call
```bash
curl -X POST https://your-server/v1/audio/transcriptions \
  -F 'file=@audio.wav' \
  -F 'response_format=verbose_json' \
  -F 'output_bucket=my-results-bucket' \
  -F 'output_use_r2=true'
  # Results saved to cloud storage AND returned inline
```

**Result**: Transcription comes back inline PLUS storage confirmation
```json
{
  "status": "success",
  "text": "Transcribed text...", 
  "result": {
    "segments": [...]
  },
  "storage_info": {
    "uploaded": true,
    "storage_url": "s3://my-results-bucket/transcriptions/20241201_result.json",
    "service": "R2"
  }
}
```

## **Parameter Details**

| Parameter | Purpose | Values | Required |
|-----------|---------|---------|----------|
| `output_bucket` | Target storage bucket | Any bucket name | No - triggers storage export |
| `output_key` | Custom object key | Custom path/filename | No - auto-generated if not provided |
| `output_use_r2` | Storage service choice | `true` (R2) / `false` (S3) | No - defaults to `false` (S3) |

## **Auto-Generated Keys**

If you specify `output_bucket` but no `output_key`, the server auto-generates:
```
transcriptions/{timestamp}_result.json

Examples:
- transcriptions/20241201_143022_123456_result.json
- transcriptions/20241201_150430_789012_result.json
```

## **Frontend Integration Examples**

### **JavaScript - Inline Results Only**
```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('response_format', 'verbose_json');

const response = await fetch('/v1/audio/transcriptions', {
  method: 'POST',
  body: formData
});

const result = await response.json();
// Use result.text or result.result immediately
```

### **JavaScript - With Cloud Storage Export**
```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('response_format', 'verbose_json');
formData.append('output_bucket', 'customer-transcriptions');
formData.append('output_use_r2', 'true');

const response = await fetch('/v1/audio/transcriptions', {
  method: 'POST', 
  body: formData
});

const result = await response.json();
// Use result.text immediately AND
// Access result.storage_info.storage_url for the archived version
```

### **JavaScript - Custom Storage Path**
```javascript
const transcriptionId = generateUniqueId();
const formData = new FormData();
formData.append('file', audioFile);
formData.append('output_bucket', 'enterprise-transcriptions');
formData.append('output_key', `meetings/2024/december/${transcriptionId}.json`);
formData.append('output_use_r2', 'false'); // Use S3

const response = await fetch('/v1/audio/transcriptions', {
  method: 'POST',
  body: formData
});
```

## **Use Case Recommendations**

### **Inline Results (No output_bucket)**
✅ **Good for:**
- Real-time chat/voice assistants
- Small transcriptions (<100KB results)
- Immediate display to users
- Simple integrations

### **Cloud Storage Export (With output_bucket)**  
✅ **Good for:**
- Meeting transcriptions (archival)
- Batch processing workflows  
- Large transcriptions (>1MB results)
- Compliance/audit requirements
- Downstream AI processing
- Analytics and reporting

## **Storage Service Choice**

### **R2 Export** (`output_use_r2=true`)
- **Cost**: Very low storage costs
- **Performance**: Fast for your existing R2 setup
- **Use when**: You want to keep everything in R2 ecosystem

### **S3 Export** (`output_use_r2=false`)
- **Cost**: Standard S3 pricing  
- **Performance**: Excellent global performance
- **Use when**: Customer has existing S3 workflows
- **Requires**: AWS credentials configured on server

## **Summary**

Users specify result destinations by adding these optional parameters to their **existing** `/v1/audio/transcriptions` call:

- **No parameters** = Inline results only
- **`output_bucket=my-bucket`** = Results saved to cloud storage + returned inline
- **`output_use_r2=true/false`** = Choose R2 or S3 for storage

**Simple and flexible** - works with any input source (direct upload, R2, or S3 pre-signed URLs)! 