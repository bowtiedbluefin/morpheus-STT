# WhisperX Intelligent Load Balancer

A custom load balancer that routes transcription requests to the backend instance with the **fewest active jobs**. Supports up to 10 backend instances configured via environment variables.

## Features

- ✅ **Intelligent Routing**: Routes to instance with fewest `active_jobs`
- ✅ **Health Monitoring**: Continuous health checks on all instances
- ✅ **Automatic Failover**: Removes unhealthy instances from rotation
- ✅ **Flexible Configuration**: Supports 1-10 instances via environment variables
- ✅ **Pass-through Responses**: Returns exact backend responses to clients
- ✅ **Production Ready**: Docker support, health checks, logging

## How It Works

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /v1/audio/transcriptions
       ▼
┌─────────────────────────────┐
│   Load Balancer (Port 8080) │
│                             │
│  1. Check /health on each   │
│     instance for active_jobs│
│                             │
│  2. Select instance with    │
│     fewest active_jobs      │
│                             │
│  3. Proxy request           │
│                             │
│  4. Return exact response   │
└─────────────────────────────┘
       │
   ┌───┴────┬──────┬──────┐
   ▼        ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────┐ ...
│Inst 1│ │Inst 2│ │Inst 3│ (up to 10)
│2 jobs│ │5 jobs│ │1 job │
└──────┘ └──────┘ └──────┘
                    ▲
                    │
              Selected (fewest jobs)
```

## Quick Start

### 1. Configure Environment

```bash
cd load-balancer

# Copy example env file
cp .env.example .env

# Edit with your instance URLs
nano .env
```

Set your Akash instance URLs:
```bash
INSTANCE_1_URL=https://instance1.provider.akash.network
INSTANCE_2_URL=https://instance2.provider.akash.network
INSTANCE_3_URL=https://instance3.provider.akash.network
```

### 2. Run with Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 3. Test

```bash
# Health check
curl http://localhost:8080/health

# Transcription test
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -F "file=@test-audio.wav" \
  -F "response_format=json"
```

## Alternative: Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export INSTANCE_1_URL=https://instance1.provider.akash.network
export INSTANCE_2_URL=https://instance2.provider.akash.network
export INSTANCE_3_URL=https://instance3.provider.akash.network

# Run
python load_balancer.py
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Load balancer port | `8080` | No |
| `HEALTH_CHECK_INTERVAL` | Health check frequency (seconds) | `30` | No |
| `REQUEST_TIMEOUT` | Max request time (seconds) | `3600` | No |
| `INSTANCE_1_URL` | First backend instance URL | - | **Yes** |
| `INSTANCE_2_URL` | Second backend instance URL | - | No |
| `INSTANCE_3_URL` | Third backend instance URL | - | No |
| ... | ... | - | No |
| `INSTANCE_10_URL` | Tenth backend instance URL | - | No |

**Note**: At least `INSTANCE_1_URL` must be configured. The load balancer will automatically detect and use all configured instances (1-10).

## API Endpoints

### `GET /`
Root endpoint with service information.

**Response:**
```json
{
  "service": "WhisperX Load Balancer",
  "version": "1.0.0",
  "instances": 3,
  "endpoints": {
    "health": "/health",
    "transcription": "/v1/audio/transcriptions"
  }
}
```

### `GET /health`
Load balancer health check. Returns aggregated status of all backend instances.

**Response:**
```json
{
  "total_instances": 3,
  "healthy_instances": 3,
  "total_active_jobs": 7,
  "instances": [
    {
      "url": "https://instance1...",
      "is_healthy": true,
      "active_jobs": 2,
      "last_check": "2025-11-01T12:00:00",
      "consecutive_failures": 0,
      "error_message": null
    },
    {
      "url": "https://instance2...",
      "is_healthy": true,
      "active_jobs": 5,
      "last_check": "2025-11-01T12:00:00",
      "consecutive_failures": 0,
      "error_message": null
    },
    {
      "url": "https://instance3...",
      "is_healthy": true,
      "active_jobs": 0,
      "last_check": "2025-11-01T12:00:00",
      "consecutive_failures": 0,
      "error_message": null
    }
  ]
}
```

### `POST /v1/audio/transcriptions`
Main transcription endpoint. Accepts the same parameters as WhisperX API.

Routes request to the instance with the **fewest active_jobs**.

**Parameters:**
- `file` (required): Audio file to transcribe
- `language`: Language code (e.g., "en", "es")
- `response_format`: Output format ("json", "text", "srt", "vtt", "verbose_json")
- `enable_diarization`: Enable speaker diarization (true/false)
- `min_speakers`, `max_speakers`: Speaker count constraints
- `optimized_alignment`: Use WAV2VEC2 alignment (true/false)
- `batch_size`: Processing batch size
- And all other WhisperX parameters

**Returns:** Exact response from the backend instance.

### `POST /transcribe`
Legacy alias for `/v1/audio/transcriptions`.

## Health Check Logic

1. **Every 30 seconds** (configurable), the load balancer checks `/health` on each instance
2. Extracts `active_jobs` from the health response
3. Marks instance unhealthy after **3 consecutive failures**
4. Healthy instances are considered for routing

## Routing Algorithm

When a transcription request arrives:

1. Get list of **healthy instances**
2. If no healthy instances → return `503 Service Unavailable`
3. **Select instance with minimum `active_jobs`**
4. Proxy the request to that instance
5. Return the exact response to the client

## Monitoring

### View Logs

```bash
# Docker Compose
docker-compose logs -f

# Local
# Logs go to stdout
```

### Check Instance Status

```bash
curl http://localhost:8080/health | jq
```

### Log Levels

The load balancer logs:
- **INFO**: Startup, routing decisions, instance selections
- **WARNING**: Instance failures, unhealthy instances
- **ERROR**: Request failures, no healthy instances
- **DEBUG**: Health check details (enable with log level config)

## Production Deployment

### With Docker Compose

```bash
# Production docker-compose.yml
version: '3.8'
services:
  load-balancer:
    build: .
    ports:
      - "8080:8080"
    env_file:
      - .env
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### With systemd (Linux)

Create `/etc/systemd/system/whisperx-lb.service`:
```ini
[Unit]
Description=WhisperX Load Balancer
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/whisperx-load-balancer
EnvironmentFile=/opt/whisperx-load-balancer/.env
ExecStart=/usr/bin/python3 /opt/whisperx-load-balancer/load_balancer.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable whisperx-lb
sudo systemctl start whisperx-lb
sudo systemctl status whisperx-lb
```

## SSL/TLS Setup

Use a reverse proxy (nginx) for SSL:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Large file support
        client_max_body_size 4096M;
        proxy_read_timeout 3600s;
        proxy_request_buffering off;
    }
}
```

## Scaling

### Add More Instances

Simply add more environment variables:

```bash
# .env
INSTANCE_4_URL=https://instance4.provider.akash.network
INSTANCE_5_URL=https://instance5.provider.akash.network
```

Restart:
```bash
docker-compose restart
```

The load balancer will automatically detect and use the new instances.

### Remove Instances

Remove or comment out the environment variable and restart:

```bash
# INSTANCE_3_URL=https://instance3.provider.akash.network  # Commented out
```

## Troubleshooting

### No instances configured error

```
ValueError: At least one instance endpoint must be configured
```

**Solution**: Set at least `INSTANCE_1_URL` in your `.env` file or environment.

### All instances marked unhealthy

Check the `/health` endpoint on your instances directly:
```bash
curl https://instance1.provider.akash.network/health
```

Ensure they return a valid JSON response with `active_jobs` field.

### Requests timing out

Increase `REQUEST_TIMEOUT`:
```bash
REQUEST_TIMEOUT=7200  # 2 hours
```

### High latency

Check `/health` to see active job distribution. If one instance has many jobs, add more instances.

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest httpx

# Run tests
pytest test_load_balancer.py
```

### Code Structure

```
load-balancer/
├── load_balancer.py       # Main application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image
├── docker-compose.yml    # Docker Compose config
├── .env.example          # Example environment config
└── README.md            # This file
```

## License

Same as parent WhisperX project.

## Support

For issues with:
- **Load balancer**: Check logs, verify instance URLs
- **Backend instances**: Check Akash deployment, instance health
- **Akash deployment**: Refer to Akash documentation

