#!/bin/bash

# 🚗 Parking Segmentation System - Quick Deploy Script

echo "🚗 Parking Segmentation System - Deployment Script"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+"
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if camera is accessible
echo "📹 Testing camera access..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ Camera is accessible')
    cap.release()
else:
    print('❌ Camera is not accessible')
    exit(1)
"

# Test the application locally
echo "🧪 Testing application locally..."
python3 ParkingSegmentation_FastAPI.py &
APP_PID=$!

# Wait for app to start
sleep 5

# Test endpoints
echo "🔍 Testing endpoints..."
curl -s http://localhost:8000/health > /dev/null && echo "✅ Health check passed" || echo "❌ Health check failed"
curl -s http://localhost:8000/parking_status > /dev/null && echo "✅ Status endpoint working" || echo "❌ Status endpoint failed"

# Kill the test app
kill $APP_PID 2>/dev/null

echo ""
echo "🎉 Local testing completed!"
echo ""
echo "🌐 Deployment Options:"
echo "1. Railway.app (Recommended - Free)"
echo "2. Render.com (Free tier available)"
echo "3. Heroku (Paid)"
echo "4. Google Cloud Run (Pay per use)"
echo "5. Docker (Local/Cloud)"
echo ""
echo "📚 See DEPLOYMENT_GUIDE.md for detailed instructions"
echo ""
echo "🚀 Quick Railway deployment:"
echo "1. Install Railway CLI: npm install -g @railway/cli"
echo "2. Login: railway login"
echo "3. Deploy: railway up"
echo ""
echo "🐳 Quick Docker deployment:"
echo "docker-compose up --build" 