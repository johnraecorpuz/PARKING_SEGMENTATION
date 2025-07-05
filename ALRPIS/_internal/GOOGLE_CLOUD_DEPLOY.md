# 🚀 Google Cloud Run Deployment (No CLI Required)

## Step 1: Prepare Your Code
1. Make sure all files are in your GitHub repository
2. Ensure `Dockerfile` is in the root directory
3. Verify `requirements.txt` is present

## Step 2: Access Google Cloud Console
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project or select existing one
3. Enable Cloud Run API

## Step 3: Deploy via Web Interface
1. Go to **Cloud Run** in the left menu
2. Click **"Create Service"**
3. Choose **"Deploy one revision from an existing container image"**
4. Click **"Browse"** and select your GitHub repository
5. Set build configuration:
   - **Source**: GitHub
   - **Repository**: Your repo
   - **Branch**: main
   - **Dockerfile**: `./Dockerfile`

## Step 4: Configure Service
- **Service name**: `parking-segmentation`
- **Region**: Choose closest to you
- **CPU allocation**: 1 CPU
- **Memory**: 2 GB
- **Maximum number of instances**: 10
- **Port**: 8000

## Step 5: Set Environment Variables
Add these in the **Variables & Secrets** section:
```
OPENCV_LOG_LEVEL=ERROR
PYTHONUNBUFFERED=1
```

## Step 6: Deploy
1. Click **"Create"**
2. Wait for build and deployment (5-10 minutes)
3. Get your URL: `https://your-service-hash-uc.a.run.app`

## Step 7: Custom Domain (Optional)
1. Go to **Cloud Run** → Your service
2. Click **"Manage Custom Domains"**
3. Add your domain
4. Update DNS records as instructed

## Cost Estimate
- **Free tier**: 2 million requests/month
- **After free tier**: ~$0.40 per million requests
- **Very cost-effective** for most use cases 