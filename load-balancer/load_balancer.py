#!/usr/bin/env python3
"""
WhisperX Intelligent Load Balancer
===================================
Routes transcription requests to the instance with the fewest active jobs.
Supports up to 10 backend instances configured via environment variables.
"""

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PORT = int(os.getenv('PORT', '8080'))
HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))  # seconds
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '86400'))  # 24 hour default

# Load instance endpoints from environment variables
# INSTANCE_1_URL, INSTANCE_2_URL, ... INSTANCE_10_URL
INSTANCE_ENDPOINTS = []
for i in range(1, 11):
    endpoint = os.getenv(f'INSTANCE_{i}_URL')
    if endpoint:
        # Normalize URL - add http:// if no protocol specified
        if not endpoint.startswith(('http://', 'https://')):
            endpoint = f'http://{endpoint}'
        INSTANCE_ENDPOINTS.append(endpoint.rstrip('/'))

if not INSTANCE_ENDPOINTS:
    logger.error("No instance endpoints configured! Set INSTANCE_1_URL, INSTANCE_2_URL, etc.")
    raise ValueError("At least one instance endpoint must be configured")

logger.info(f"Configured {len(INSTANCE_ENDPOINTS)} backend instances:")
for i, endpoint in enumerate(INSTANCE_ENDPOINTS, 1):
    logger.info(f"  Instance {i}: {endpoint}")


@dataclass
class InstanceHealth:
    """Health status of a backend instance"""
    url: str
    is_healthy: bool = True
    active_transcriptions: int = 0  # Total load: queued + running transcriptions
    active_jobs: int = 0  # Legacy: async jobs only (kept for backward compatibility)
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    pending_requests: int = 0  # Track requests we've routed but not yet completed


class LoadBalancer:
    """Intelligent load balancer that routes to instance with fewest active jobs"""
    
    def __init__(self, instance_urls: List[str]):
        self.instances = {url: InstanceHealth(url=url) for url in instance_urls}
        self.health_check_task = None
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        
    async def start(self):
        """Start background health checking"""
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Load balancer started with health checking enabled")
        
    async def stop(self):
        """Stop background tasks and cleanup"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()
        logger.info("Load balancer stopped")
        
    async def _health_check_loop(self):
        """Background task to continuously check instance health"""
        while True:
            try:
                await self._check_all_instances()
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
                
    async def _check_all_instances(self):
        """Check health of all instances in parallel"""
        tasks = [self._check_instance_health(url) for url in self.instances.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _check_instance_health(self, instance_url: str):
        """Check health of a single instance and update its status"""
        instance = self.instances[instance_url]
        
        try:
            health_url = f"{instance_url}/health"
            response = await self.client.get(health_url, timeout=5.0)
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Extract metrics from health response
                # Prioritize active_transcriptions (total load) over active_jobs (async only)
                concurrency_data = health_data.get('concurrency', {})
                active_transcriptions = concurrency_data.get('active_transcriptions', 0)
                active_jobs = concurrency_data.get('active_jobs', 0)
                
                # Update instance health
                instance.is_healthy = True
                instance.active_transcriptions = active_transcriptions
                instance.active_jobs = active_jobs
                instance.last_check = datetime.now()
                instance.consecutive_failures = 0
                instance.error_message = None
                
                logger.debug(f"{instance_url}: healthy, {active_transcriptions} active transcriptions, {active_jobs} async jobs")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            instance.consecutive_failures += 1
            instance.error_message = str(e)
            
            # Mark unhealthy after 3 consecutive failures
            if instance.consecutive_failures >= 3:
                instance.is_healthy = False
                logger.warning(
                    f"{instance_url}: marked unhealthy after {instance.consecutive_failures} "
                    f"failures. Error: {instance.error_message}"
                )
            else:
                logger.debug(f"{instance_url}: health check failed ({instance.consecutive_failures}/3): {e}")
                
    def get_best_instance(self) -> Optional[str]:
        """
        Get the instance with the fewest active transcriptions.
        Combines actual active_transcriptions (queued + running) from health checks 
        with pending_requests to handle simultaneous requests properly.
        Returns None if no healthy instances are available.
        """
        healthy_instances = [
            (url, instance) 
            for url, instance in self.instances.items() 
            if instance.is_healthy
        ]
        
        if not healthy_instances:
            logger.error("No healthy instances available!")
            return None
            
        # Sort by total load: active_transcriptions (queued + running) + pending_requests (tracked locally)
        # This ensures simultaneous requests get distributed across instances
        best_instance = min(
            healthy_instances, 
            key=lambda x: x[1].active_transcriptions + x[1].pending_requests
        )
        
        logger.info(
            f"Selected {best_instance[0]} with {best_instance[1].active_transcriptions} active transcriptions "
            f"+ {best_instance[1].pending_requests} pending requests"
        )
        
        return best_instance[0]
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all instance health statuses"""
        return {
            "total_instances": len(self.instances),
            "healthy_instances": sum(1 for i in self.instances.values() if i.is_healthy),
            "total_active_transcriptions": sum(i.active_transcriptions for i in self.instances.values()),
            "total_active_jobs": sum(i.active_jobs for i in self.instances.values()),
            "total_pending_requests": sum(i.pending_requests for i in self.instances.values()),
            "instances": [
                {
                    "url": url,
                    "is_healthy": inst.is_healthy,
                    "active_transcriptions": inst.active_transcriptions,
                    "active_jobs": inst.active_jobs,
                    "pending_requests": inst.pending_requests,
                    "total_load": inst.active_transcriptions + inst.pending_requests,
                    "last_check": inst.last_check.isoformat() if inst.last_check else None,
                    "consecutive_failures": inst.consecutive_failures,
                    "error_message": inst.error_message
                }
                for url, inst in self.instances.items()
            ]
        }
        
    async def proxy_request(
        self,
        request: Request,
        path: str = "/v1/audio/transcriptions"
    ) -> Response:
        """
        Proxy request to the best available instance.
        Pure pass-through - sends exact request and returns exact response.
        Tracks pending requests to distribute simultaneous requests properly.
        """
        # Get best instance
        instance_url = self.get_best_instance()
        
        if not instance_url:
            raise HTTPException(
                status_code=503,
                detail="No healthy backend instances available"
            )
        
        # Increment pending requests counter for this instance
        self.instances[instance_url].pending_requests += 1
        
        try:
            # Get the raw request body
            body = await request.body()
            
            # Get all headers except host
            headers = dict(request.headers)
            headers.pop('host', None)
            
            # Make request to backend instance
            endpoint_url = f"{instance_url}{path}"
            
            logger.info(f"Proxying {request.method} request to {endpoint_url} (body size: {len(body)} bytes)")
            
            response = await self.client.request(
                method=request.method,
                url=endpoint_url,
                content=body,
                headers=headers
            )
            
            # Return exact response from backend
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get('content-type')
            )
            
        except httpx.TimeoutException:
            logger.error(f"Request to {instance_url} timed out")
            raise HTTPException(
                status_code=504,
                detail="Backend request timed out"
            )
        except Exception as e:
            logger.error(f"Error proxying request to {instance_url}: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Backend request failed: {str(e)}"
            )
        finally:
            # Decrement pending requests counter when done
            self.instances[instance_url].pending_requests -= 1


# Initialize FastAPI app
app = FastAPI(
    title="WhisperX Load Balancer",
    description="Intelligent load balancer for WhisperX transcription instances",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize load balancer
load_balancer = LoadBalancer(INSTANCE_ENDPOINTS)


@app.on_event("startup")
async def startup_event():
    """Start load balancer on application startup"""
    await load_balancer.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop load balancer on application shutdown"""
    await load_balancer.stop()


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "WhisperX Load Balancer",
        "version": "1.0.0",
        "instances": len(INSTANCE_ENDPOINTS),
        "endpoints": {
            "health": "/health",
            "transcription": "/v1/audio/transcriptions"
        }
    }


@app.get("/health")
async def health_check():
    """
    Load balancer health check endpoint.
    Returns aggregated health status of all backend instances.
    """
    summary = load_balancer.get_health_summary()
    
    # Return 503 if no healthy instances
    status_code = 200 if summary["healthy_instances"] > 0 else 503
    
    return JSONResponse(
        content=summary,
        status_code=status_code
    )


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    """
    Main transcription endpoint.
    Pure pass-through proxy - sends exact request to backend with fewest active jobs.
    """
    return await load_balancer.proxy_request(request, "/v1/audio/transcriptions")


@app.post("/transcribe")
async def transcribe_legacy(request: Request):
    """
    Legacy transcription endpoint (alias).
    Pure pass-through proxy - sends exact request to backend with fewest active jobs.
    """
    return await load_balancer.proxy_request(request, "/transcribe")


if __name__ == "__main__":
    logger.info(f"Starting WhisperX Load Balancer on port {PORT}")
    logger.info(f"Health check interval: {HEALTH_CHECK_INTERVAL}s")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT}s")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

