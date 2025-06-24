# Deploy Whisper API on Akash Network via Console UI

Simple guide to deploy your Whisper Transcription API using the user-friendly Akash Console web interface.

## Prerequisites

1. **Docker Hub account** - To host your container image
2. **Keplr wallet** with AKT tokens (15+ AKT recommended)
3. **Basic familiarity with Docker**

⚠️ **Important**: Your Docker image is ~15GB, so initial deployment will take longer for image download.

## Step 1: Build and Push Docker Image

```bash
# Build the Docker image (replace 'yourusername' with your Docker Hub username)
docker build -t yourusername/whisper-transcription-api:latest .

# Push to Docker Hub (this will take time due to 15GB size)
docker push yourusername/whisper-transcription-api:latest
```

## Step 2: Update Deployment Files

Edit both `deploy.yaml` and `deploy-gpu.yaml` files:
- Replace `your-dockerhub-username` with your actual Docker Hub username

## Step 3: Deploy via Akash Console

1. **Visit**: https://console.akash.network/
2. **Connect**: Your Keplr wallet
3. **Click**: "Deploy" button
4. **Choose**: "Build your template" or "Upload SDL"
5. **Upload**: Your `deploy.yaml` file (or `deploy-gpu.yaml` for GPU)
6. **Review**: Configuration and estimated costs
7. **Create Deployment**: Confirm transaction in Keplr
8. **Wait**: For provider bids (usually 1-3 minutes)
9. **Accept Bid**: Choose a provider and create lease
10. **Wait**: For image download (~15GB - may take 10-30 minutes)
11. **Access**: Your API via the provided URI

## Deployment Files Explained

### `deploy.yaml` (CPU Only - Cheaper)
```yaml
- CPU: 2.0 units
- Memory: 8 GiB  
- Storage: 25 GiB persistent (increased for 15GB image + 3GB model)
- Cost: ~75-150 AKT/month
```

### `deploy-gpu.yaml` (GPU Enabled - Faster)
```yaml
- CPU: 4.0 units
- Memory: 16 GiB
- GPU: 1 unit (NVIDIA RTX/A6000/V100)
- Storage: 30 GiB persistent (increased for 15GB image + 3GB model)
- Cost: ~300-700 AKT/month
```

## Important Configuration Details

### Port Mapping
- **Internal**: Your app runs on port 3333
- **External**: Exposed as port 80 globally
- **Access**: Via HTTPS URL provided by Akash

### Persistent Storage
- **Path**: `/app/models` (where Whisper models are stored)
- **Size**: 25-30 GiB (accommodates 15GB Docker image + 3GB Whisper model + working space)
- **Purpose**: Prevents re-downloading large files on restarts

### Environment Variables
- `PYTHONUNBUFFERED=1` - For proper logging
- `CUDA_VISIBLE_DEVICES=""` - CPU-only mode (GPU version removes this)

## After Deployment

Your API will be available at a URL like: `https://abc123.provider.akash.network`

**⏱️ Initial startup time**: 15-45 minutes due to:
- 15GB Docker image download
- 3GB Whisper model download on first transcription

### Key Endpoints:
- `GET /` - API status
- `POST /v1/audio/transcriptions` - Main transcription endpoint (OpenAI compatible)
- `GET /docs` - Interactive API documentation (Swagger UI)

### Test Your Deployment:
```bash
# Basic test
curl https://your-deployment-url.akash.network/

# Transcription test (replace with your URL and audio file)
curl -X POST "https://your-deployment-url.akash.network/v1/audio/transcriptions" \
  -F "file=@test-audio.wav" \
  -F "model=whisper-1"
```

## Cost Management

- **Monitor**: Usage via Akash Console dashboard
- **Estimate**: CPU deployment ~$10-25/month, GPU ~$40-100/month (varies with AKT price)
- **Control**: Close deployment anytime to stop costs
- **Note**: Higher costs due to increased storage requirements

## Troubleshooting

### If deployment fails:
1. Check Docker image is public and accessible
2. Verify sufficient AKT balance (need more due to larger storage)
3. Try different provider if bid acceptance fails
4. **Be patient**: 15GB image download takes time

### If service doesn't start:
1. Check logs in Akash Console
2. Verify image name matches exactly
3. Ensure port 3333 is exposed in your app
4. **Wait longer**: Large image needs time to download and start

### Performance Notes:
- **First deployment**: Takes 15-45 minutes (image + model download)
- **Subsequent restarts**: Faster if using persistent storage
- **First transcription**: Additional 5-10 minutes for model download
- **Subsequent requests**: Fast as model is cached
- **GPU version**: 5-10x faster transcription than CPU

## Optimization Tips

### Reduce Image Size (Optional):
```dockerfile
# Use multi-stage build to reduce image size
FROM python:3.11-slim as builder
# ... build dependencies

FROM python:3.11-slim as runtime
# Copy only necessary files from builder
```

### Alternative: Use smaller base images
- Consider `python:3.11-alpine` for smaller base
- Remove unnecessary packages after installation

## Security Considerations

⚠️ **Important**: This deployment exposes your API globally without authentication.

For production use, consider:
- Adding API key authentication
- Implementing rate limiting
- Monitoring usage and costs
- Using HTTPS only (Akash provides this automatically)

## Support

- **Akash Console**: Built-in help and documentation
- **Community**: [Akash Discord](https://discord.akash.network)
- **Issues**: Check container logs via Console interface 