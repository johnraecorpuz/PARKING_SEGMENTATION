# 🚗 Parking Segmentation System

A real-time parking space detection and monitoring system using computer vision and machine learning.

## 🌟 Features

- **Real-time video processing** with OpenCV
- **AI-powered parking detection** using YOLO models
- **Person detection** for security monitoring
- **Web interface** with live video streaming
- **REST API** for parking status
- **Mobile-friendly** responsive design
- **Cloud deployment ready** with multiple platform support

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam or camera device
- YOLO model files (included in Models/ directory)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/parking-segmentation.git
cd parking-segmentation
```

2. **Install dependencies**
```bash
pip install -r ALRPIS/_internal/requirements.txt
```

3. **Run the application**
```bash
# FastAPI version (recommended)
python ALRPIS/_internal/ParkingSegmentation_FastAPI.py

# Flask version
python ALRPIS/_internal/ParkingSegmentation.py
```

4. **Access the application**
- Web UI: http://localhost:8000
- Video Feed: http://localhost:8000/video_feed
- API Docs: http://localhost:8000/docs
- Parking Status: http://localhost:8000/parking_status

## 📁 Project Structure

```
PARKING_SEG/
├── ALRPIS/
│   └── _internal/
│       ├── ParkingSegmentation.py          # Flask version
│       ├── ParkingSegmentation_FastAPI.py  # FastAPI version (recommended)
│       ├── Models/                          # YOLO model files
│       │   ├── best.pt                      # Parking detection model
│       │   └── yolov8m.pt                   # Object detection model
│       ├── requirements.txt                 # Python dependencies
│       ├── Dockerfile                       # Docker configuration
│       ├── docker-compose.yml              # Docker Compose setup
│       ├── render.yaml                      # Render.com deployment
│       ├── railway.json                     # Railway.app deployment
│       ├── DEPLOYMENT_GUIDE.md             # Deployment instructions
│       ├── GOOGLE_CLOUD_DEPLOY.md          # Google Cloud deployment
│       └── deploy_no_nodejs.py             # Deployment helper script
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
```

## 🌐 Deployment Options

### **1. Render.com (Recommended - Free)**
- Free tier available
- Custom domain support
- Automatic HTTPS
- Easy web interface deployment

### **2. Railway.app (Free)**
- Free tier available
- Automatic deployments
- Custom domains
- Great developer experience

### **3. Google Cloud Run**
- Pay per use
- Highly scalable
- Custom domains
- Enterprise-grade

### **4. Heroku**
- Classic choice
- Custom domains
- Paid plans

### **5. Docker (Local/Cloud)**
```bash
# Build and run with Docker
cd ALRPIS/_internal
docker-compose up --build
```

## 📊 API Endpoints

### FastAPI Version
- `GET /` - Web interface
- `GET /video_feed` - Live video stream
- `GET /parking_status` - JSON parking status
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

### Response Format
```json
{
  "parked_cars": 5,
  "available_spaces": 3
}
```

## 🔧 Configuration

### Environment Variables
```bash
OPENCV_LOG_LEVEL=ERROR
PYTHONUNBUFFERED=1
```

### Camera Settings
- Default resolution: 1920x1080
- Frame rate: 30 FPS
- Camera index: 0 (auto-detects)

## 🛠️ Development

### Running Tests
```bash
# Test camera access
python ALRPIS/_internal/deploy_no_nodejs.py

# Test application
python -c "from ALRPIS._internal.ParkingSegmentation_FastAPI import app; print('✅ App loaded successfully')"
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📱 Mobile Access

The application is fully responsive and works on:
- **iOS Safari**
- **Android Chrome**
- **Progressive Web App** (add to home screen)

## 🔒 Security

- HTTPS enabled on all cloud deployments
- Rate limiting available
- CORS configuration for production
- Environment variable protection

## 📈 Performance

- **Video streaming**: Optimized for low latency
- **Model inference**: CPU-optimized
- **Memory usage**: Efficient frame processing
- **Network**: Compressed JPEG streaming

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLO models
- **OpenCV** for computer vision
- **FastAPI** for modern web framework
- **Flask** for lightweight web framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/parking-segmentation/issues)
- **Documentation**: [DEPLOYMENT_GUIDE.md](ALRPIS/_internal/DEPLOYMENT_GUIDE.md)
- **Email**: your-email@example.com

## 🚀 Quick Deploy

Want to deploy instantly? Use our deployment helper:

```bash
python ALRPIS/_internal/deploy_no_nodejs.py
```

This will guide you through the deployment process step by step!

---

⭐ **Star this repository if you find it helpful!** 