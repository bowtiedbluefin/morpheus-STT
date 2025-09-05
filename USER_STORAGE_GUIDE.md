# User Storage Guide: Save Transcriptions to Your Own S3 Bucket

## Overview
There are several ways users can save transcription results to their own storage:

## 🎯 Method 1: Pre-signed Upload URLs (Recommended)

### Step 1: Generate Upload URL
```bash
# User generates upload URL on their side
aws s3 presign s3://my-bucket/transcriptions/result-$(date +%s).json \
  --expires-in 3600 \
  --method PUT
```

### Step 2: Use Upload URL with API
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "upload_presigned_url=https://my-bucket.s3.amazonaws.com/transcriptions/result.json?X-Amz-Signature=..." \
  -F "response_format=json"
```

**Pros:**
- ✅ Most secure (no credentials exposed)
- ✅ User maintains full control 
- ✅ Works with any S3-compatible service
- ✅ No server-side AWS setup needed

## 🔐 Method 2: S3 Bucket Policies (Cross-Account Access)

### Step 1: Configure Your Bucket Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowWhisperAPIWrite",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::SERVER-ACCOUNT:user/whisper-api"
      },
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl"
      ],
      "Resource": "arn:aws:s3:::YOUR-BUCKET/transcriptions/*"
    }
  ]
}
```

### Step 2: Use Standard API
```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "output_bucket=your-bucket-name" \
  -F "stored_output=false" \
  -F "response_format=json"
```

## 📋 Method 3: User-Provided Credentials (Not Recommended)

For completeness, you could extend the API to accept user credentials:

```bash
curl -X POST "http://localhost:3333/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "user_aws_key=AKIAEXAMPLE" \
  -F "user_aws_secret=secretkey" \
  -F "output_bucket=user-bucket" \
  -F "aws_region=us-east-2"
```

**⚠️ Security Risk:** Credentials transmitted in requests.

## 🏆 Recommended Architecture

For production SaaS:

1. **User generates upload URLs** client-side
2. **API accepts upload URLs** as parameter
3. **Results uploaded directly** to user's bucket
4. **No credentials** ever touch your server

### Example Client Implementation (JavaScript):
```javascript
// 1. Generate upload URL
const uploadUrl = await generatePresignedUrl('my-bucket', 'transcriptions/result.json');

// 2. Call transcription API
const response = await fetch('/v1/audio/transcriptions', {
  method: 'POST',
  body: formData.append('upload_presigned_url', uploadUrl)
});

// 3. User's file is automatically saved to their bucket!
```

## 🔧 API Parameters Summary

| Parameter | Description | Example |
|-----------|-------------|---------|
| `upload_presigned_url` | Pre-signed upload URL for user's bucket | `https://my-bucket.s3.amazonaws.com/...` |
| `output_bucket` | Server-managed bucket name | `shared-transcriptions` |
| `stored_output` | `true`=R2, `false`=S3 | `false` |

**Priority:** `upload_presigned_url` > `output_bucket` 