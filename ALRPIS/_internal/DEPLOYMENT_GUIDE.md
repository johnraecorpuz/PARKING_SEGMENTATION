# 🚗 Parking Segmentation System - Deployment Guide

This guide covers deploying your parking segmentation system to various cloud platforms with permanent domains.

## 🌟 **Recommended: FastAPI + Cloudflare**

### 1. **Railway.app** (Easiest)
- **Cost**: Free tier available
- **Domain**: Automatic subdomain + custom domain support
- **Deployment**: Git-based

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Initialize project
railway init

# 4. Deploy
railway up
```

### 2. **Render.com** (Great Free Tier)
- **Cost**: Free tier with limitations
- **Domain**: Automatic subdomain + custom domain
- **Deployment**: Git-based

```yaml
# render.yaml
services:
  - type: web
    name: parking-segmentation
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn ParkingSegmentation_FastAPI:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### 3. **Heroku** (Classic Choice)
- **Cost**: Free tier discontinued, paid plans
- **Domain**: Automatic subdomain + custom domain
- **Deployment**: Git-based

```bash
# 1. Install Heroku CLI
# 2. Create Procfile
echo "web: uvicorn ParkingSegmentation_FastAPI:app --host 0.0.0.0 --port \$PORT" > Procfile

# 3. Deploy
heroku create your-app-name
git push heroku main
```

### 4. **Google Cloud Run** (Scalable)
- **Cost**: Pay per use
- **Domain**: Custom domain support
- **Deployment**: Container-based

```bash
# 1. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/parking-segmentation

# 2. Deploy to Cloud Run
gcloud run deploy parking-segmentation \
  --image gcr.io/PROJECT_ID/parking-segmentation \
  --platform managed \
  --allow-unauthenticated \
  --port 8000
```

### 5. **AWS Lambda + API Gateway** (Serverless)
- **Cost**: Pay per request
- **Domain**: Custom domain via API Gateway
- **Deployment**: Serverless

```yaml
# serverless.yml
service: parking-segmentation

provider:
  name: aws
  runtime: python3.11

functions:
  api:
    handler: handler.api
    events:
      - http:
          path: /{proxy+}
          method: ANY
```

### 6. **DigitalOcean App Platform**
- **Cost**: Starting at $5/month
- **Domain**: Automatic subdomain + custom domain
- **Deployment**: Git-based

```yaml
# .do/app.yaml
name: parking-segmentation
services:
  - name: web
    source_dir: /
    github:
      repo: your-username/your-repo
      branch: main
    run_command: uvicorn ParkingSegmentation_FastAPI:app --host 0.0.0.0 --port $PORT
    environment_slug: python
```

## 🔧 **Local Testing with Docker**

```bash
# Build and run locally
docker-compose up --build

# Access at http://localhost:8000
```

## 🌐 **Custom Domain Setup**

### 1. **Cloudflare (Recommended)**
```bash
# 1. Add your domain to Cloudflare
# 2. Update nameservers
# 3. Add CNAME record:
#    parking.yourdomain.com -> your-app.railway.app
# 4. Enable SSL/TLS
```

### 2. **Domain Verification**
Most platforms require domain verification:
- Add TXT record: `_verification.yourdomain.com`
- Wait for verification (usually 5-10 minutes)

## 📱 **Mobile Access**

Your app will be accessible on mobile devices:
- **iOS Safari**: `https://your-domain.com`
- **Android Chrome**: `https://your-domain.com`
- **Progressive Web App**: Add to home screen

## 🔒 **Security Considerations**

### 1. **Environment Variables**
```bash
# Add to your deployment platform
OPENCV_LOG_LEVEL=ERROR
PYTHONUNBUFFERED=1
```

### 2. **Rate Limiting**
```python
# Add to your FastAPI app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

### 3. **CORS Configuration**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 **Monitoring & Analytics**

### 1. **Health Checks**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "camera": cap.isOpened(),
        "timestamp": datetime.now().isoformat()
    }
```

### 2. **Logging**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Status: {response.status_code}")
    return response
```

## 🚀 **Performance Optimization**

### 1. **Image Compression**
```python
# Reduce JPEG quality for faster streaming
ret, jpeg = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
```

### 2. **Frame Rate Control**
```python
# Limit frame rate to reduce bandwidth
await asyncio.sleep(0.1)  # 10 FPS instead of 30
```

### 3. **Caching**
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
```

## 💰 **Cost Comparison**

| Platform | Free Tier | Paid Plans | Custom Domain |
|----------|-----------|------------|---------------|
| Railway | ✅ | $5/month | ✅ |
| Render | ✅ | $7/month | ✅ |
| Heroku | ❌ | $7/month | ✅ |
| Cloud Run | ✅ | Pay per use | ✅ |
| DigitalOcean | ❌ | $5/month | ✅ |

## 🎯 **Recommended Setup**

1. **Development**: Docker Compose
2. **Staging**: Railway.app (free)
3. **Production**: Render.com or Google Cloud Run
4. **Domain**: Cloudflare DNS
5. **SSL**: Automatic via platform

## 📞 **Support**

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Uvicorn**: https://www.uvicorn.org/
- **OpenCV**: https://opencv.org/
- **Ultralytics**: https://ultralytics.com/ 