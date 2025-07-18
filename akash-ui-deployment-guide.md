# Deploy Whisper API on Akash Network (GPU Only)

Simple guide to deploy your Whisper Transcription API on the Akash Network using the user-friendly Console web interface. This guide is for GPU deployments only.

## Prerequisites

1.  **Docker Hub Account**: To host your container image.
2.  **Hugging Face Account**: To get an access token for speaker diarization.
    - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to create a token with `read` permissions.
    - You must also accept the user conditions for the diarization model here: [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3.  **Keplr Wallet**: With AKT tokens (20+ AKT recommended for GPU deployment).
4.  **Basic Familiarity with Docker**.

⚠️ **Important**: Your final Docker image will be very large (>8GB). The initial deployment will take a significant amount of time to download the image to the Akash provider.

## Step 1: Build and Push Docker Image

The `HUGGINGFACE_TOKEN` is not required to build the image. It will be provided at runtime.

```bash
# Replace 'yourusername' with your Docker Hub username
docker build -t yourusername/whisper-transcription-api:latest .

# Push to Docker Hub (this will take time)
docker push yourusername/whisper-transcription-api:latest
```

## Step 2: Update Deployment File

Edit the `deploy-gpu.yaml` file:
- Replace `your-dockerhub-username` with your actual Docker Hub username.
- Replace `your_token_here` with your actual Hugging Face token.

## Step 3: Deploy via Akash Console

1.  **Visit**: [console.akash.network](https://console.akash.network/)
2.  **Connect**: Your Keplr wallet.
3.  **Click**: "Deploy" -> "Upload SDL".
4.  **Upload**: Your `deploy-gpu.yaml` file.
5.  **Review**: The configuration and estimated costs.
6.  **Create Deployment**: Confirm the transaction in Keplr.
7.  **Wait**: For a provider to bid on your deployment (1-3 minutes).
8.  **Accept Bid**: Choose a provider and create the lease.
9.  **Wait**: For the image to download. This is the longest step and can take 15-40 minutes depending on the provider's network speed.
10. **Access**: Your API via the URI provided in the Akash Console once the service is running.

## Deployment File Explained

### `deploy-gpu.yaml` (GPU Enabled)
```yaml
- CPU: 4.0 units
- Memory: 16 GiB
- GPU: 1 unit (NVIDIA RTX/A6000/V100, etc.)
- Storage: 30 GiB persistent
- Cost: Highly variable, check estimates in Console.
```

## Important Configuration Details

### Port Mapping
- **Internal**: Your app runs on port `3333`.
- **External**: Exposed as port `80` globally and accessible via HTTPS.

### Persistent Storage
- **Path**: `/app/models` is mounted for persistent storage.
- **Purpose**: Prevents re-downloading the large Whisper and Pyannote models every time the container restarts.

### Environment Variables
- `HUGGINGFACE_TOKEN`: This must be set in your `deploy-gpu.yaml` for diarization to work.
- `PYTHONUNBUFFERED=1`: Ensures proper logging output.

## After Deployment

Your API will be available at a URL like: `https://abc123.provider.akash.network`

**⏱️ Initial startup time**: Can be long (15-45 minutes) due to:
- Large Docker image download (~8-10GB).
- Whisper and Pyannote model downloads on the first transcription request (~5GB total).

### Key Endpoints:
- `GET /` - API status.
- `GET /health` - Health check for models.
- `POST /v1/audio/transcriptions` - Main transcription endpoint.
- `GET /docs` - Interactive API documentation (Swagger UI).

### Test Your Deployment:
```bash
# Basic test (replace with your URL)
curl https://your-deployment-url.akash.network/

# Transcription test (replace with your URL and audio file)
curl -X POST "https://your-deployment-url.akash.network/v1/audio/transcriptions" \
  -F "file=@path/to/audio.wav" \
  -F "model=whisper-1"
```

## Cost Management

- **Monitor**: Your usage and costs via the Akash Console dashboard.
- **Estimate**: GPU deployments are more expensive. Check the provider bids carefully.
- **Control**: You can close the deployment at any time to stop all costs.

## Troubleshooting

### If deployment fails:
1.  Check that your Docker image on Docker Hub is public.
2.  Verify you have a sufficient AKT balance in your wallet.
3.  Ensure your `HUGGINGFACE_TOKEN` is correct in `deploy-gpu.yaml`.
4.  Try a different provider if a bid fails after accepting it.

### If service doesn't start after deployment:
1.  Check the container logs in the Akash Console. This is the most important step.
2.  Verify the image name in `deploy-gpu.yaml` matches your Docker Hub image name exactly.
3.  **Be patient**. The large image and model downloads take a significant amount of time. The service will not be available until they are complete.

## Security Considerations

⚠️ **Important**: This deployment, by default, exposes your API to the public internet without authentication.

For production use, you should consider implementing security measures such as:
- API key authentication.
- Rate limiting. 