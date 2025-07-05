#!/usr/bin/env python3
"""
🚗 Parking Segmentation System - Deployment Helper (No Node.js Required)
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check and install required dependencies"""
    print("📦 Checking dependencies...")
    
    try:
        import cv2
        print("✅ OpenCV installed")
    except ImportError:
        print("❌ OpenCV not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"])
    
    try:
        import fastapi
        print("✅ FastAPI installed")
    except ImportError:
        print("❌ FastAPI not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
    
    try:
        import ultralytics
        print("✅ Ultralytics installed")
    except ImportError:
        print("❌ Ultralytics not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"])

def test_camera():
    """Test camera access"""
    print("📹 Testing camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is accessible")
            cap.release()
            return True
        else:
            print("❌ Camera is not accessible")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_application():
    """Test the FastAPI application"""
    print("🧪 Testing application...")
    try:
        # Import and test the app
        from ParkingSegmentation_FastAPI import app
        print("✅ Application imports successfully")
        return True
    except Exception as e:
        print(f"❌ Application test failed: {e}")
        return False

def show_deployment_options():
    """Show deployment options"""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT OPTIONS (No Node.js Required)")
    print("="*60)
    
    options = [
        {
            "name": "Render.com",
            "description": "Free tier, web interface, custom domains",
            "url": "https://render.com",
            "steps": [
                "1. Go to render.com and sign up",
                "2. Connect your GitHub repository",
                "3. Create new Web Service",
                "4. Select your repo and branch",
                "5. Build Command: pip install -r requirements.txt",
                "6. Start Command: uvicorn ParkingSegmentation_FastAPI:app --host 0.0.0.0 --port $PORT",
                "7. Deploy!"
            ]
        },
        {
            "name": "Railway.app",
            "description": "Free tier, web interface, easy setup",
            "url": "https://railway.app",
            "steps": [
                "1. Go to railway.app and sign up",
                "2. Connect your GitHub repository",
                "3. Create new project",
                "4. Deploy from GitHub",
                "5. Railway will auto-detect Python",
                "6. Get your URL instantly!"
            ]
        },
        {
            "name": "Google Cloud Run",
            "description": "Pay per use, highly scalable",
            "url": "https://console.cloud.google.com",
            "steps": [
                "1. Go to console.cloud.google.com",
                "2. Create new project",
                "3. Enable Cloud Run API",
                "4. Create service from GitHub",
                "5. Select your repository",
                "6. Deploy with Dockerfile"
            ]
        },
        {
            "name": "Heroku",
            "description": "Classic choice, paid plans",
            "url": "https://heroku.com",
            "steps": [
                "1. Go to heroku.com and sign up",
                "2. Create new app",
                "3. Connect GitHub repository",
                "4. Deploy from GitHub",
                "5. Add custom domain"
            ]
        }
    ]
    
    for i, option in enumerate(options, 1):
        print(f"\n{i}. {option['name']}")
        print(f"   {option['description']}")
        print(f"   URL: {option['url']}")
        print("   Steps:")
        for step in option['steps']:
            print(f"   {step}")
    
    print(f"\n💡 RECOMMENDATION: Start with {options[0]['name']} (easiest)")
    print("   It's free, has a great web interface, and supports custom domains!")

def open_deployment_guides():
    """Open deployment guides in browser"""
    guides = [
        ("Render.com", "https://render.com/docs/deploy-flask"),
        ("Railway.app", "https://docs.railway.app/deploy/deployments"),
        ("Google Cloud Run", "https://cloud.google.com/run/docs/quickstarts/build-and-deploy"),
        ("Heroku", "https://devcenter.heroku.com/articles/getting-started-with-python")
    ]
    
    print("\n🌐 Opening deployment guides...")
    for name, url in guides:
        print(f"   Opening {name} guide...")
        webbrowser.open(url)

def main():
    """Main deployment helper"""
    print("🚗 Parking Segmentation System - Deployment Helper")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    check_dependencies()
    
    # Test camera
    if not test_camera():
        print("⚠️  Camera not accessible. This might be normal in cloud deployment.")
    
    # Test application
    if not test_application():
        print("❌ Application test failed. Please check your code.")
        return
    
    # Show deployment options
    show_deployment_options()
    
    # Ask user what they want to do
    print("\n" + "="*50)
    choice = input("What would you like to do?\n1. Open deployment guides\n2. Exit\nChoice: ")
    
    if choice == "1":
        open_deployment_guides()
    
    print("\n🎉 Good luck with your deployment!")
    print("📚 Check the DEPLOYMENT_GUIDE.md file for detailed instructions.")

if __name__ == "__main__":
    main() 