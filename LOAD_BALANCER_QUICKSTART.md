# Load Balancer Quick Start

Get your custom load balancer running in **5 minutes**.

## Prerequisites

- Docker and Docker Compose installed
- 3x Akash instance URLs (from your manual deployments)

## Steps

### 1. Navigate to Load Balancer Directory

```bash
cd load-balancer
```

### 2. Configure Instance URLs

```bash
# Copy template
cp .env.example .env

# Edit with your instance URLs
nano .env
```

Set your Akash instance URLs:

```bash
# .env
INSTANCE_1_URL=https://instance1.provider.akash.network
INSTANCE_2_URL=https://instance2.provider.akash.network
INSTANCE_3_URL=https://instance3.provider.akash.network
```

### 3. Start Load Balancer

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Test

```bash
# Health check
curl http://localhost:8080/health

# Should show all 3 instances healthy
```

### 5. Test Transcription

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -F "file=@test-audio.wav" \
  -F "response_format=json"
```

## Done!

Your load balancer is now running on **http://localhost:8080**

All requests to this endpoint will be automatically routed to the instance with the fewest active jobs.

## Next Steps

- **Add SSL**: See `PRODUCTION_SETUP.md` for nginx SSL configuration
- **Monitoring**: Check `/health` endpoint for instance status
- **Scaling**: Add more instances by adding `INSTANCE_4_URL`, etc. to `.env`

## Common Issues

### "No instance endpoints configured"
Make sure `.env` file has at least `INSTANCE_1_URL` set.

### "All instances unhealthy"
Check that your Akash instances are running:
```bash
curl https://instance1.provider.akash.network/health
```

### Need to restart?
```bash
docker-compose restart
```

### View detailed logs?
```bash
docker-compose logs -f load-balancer
```

