import cv2
from ultralytics import YOLO
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn
import threading
import signal
import sys
import os
import time
import asyncio
from typing import AsyncGenerator

app = FastAPI(title="Parking Segmentation System", version="1.0.0")

# Global flag for graceful shutdown
shutdown_flag = threading.Event()

def resource_path(relative_path):
    """Get absolute path to resource inside PyInstaller .exe"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Load the YOLO models
try:
    parking_model = YOLO(resource_path('Models/best.pt')).to('cpu')
    object_model = YOLO(resource_path('Models/yolov8m.pt')).to('cpu')
    print("YOLO models loaded successfully")
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    sys.exit(1)

# Initialize camera with retry logic
def initialize_camera():
    global cap
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Camera opened successfully with index {camera_index}")
            break
        else:
            cap.release()
    
    if not cap.isOpened():
        print("Error: Could not access any camera.")
        return False
    
    # Set camera properties
    screen_width = 1920  
    screen_height = 1080  
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    
    # Verify camera is working by reading a test frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Error: Camera opened but cannot read frames.")
        cap.release()
        return False
    
    print(f"Camera initialized successfully. Frame size: {test_frame.shape}")
    return True

# Initialize camera
if not initialize_camera():
    sys.exit(1)

# Shared data structure to store parking status
parking_status = {
    'parked_cars': 0,
    'available_spaces': 0
}

# Function to process the camera frame and update the parking status
def update_parking_status():
    global parking_status, cap
    
    if not cap.isOpened():
        print("Error: Camera is not opened.")
        return None
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture an image.")
        return None

    try:
        resized_frame = cv2.resize(frame, (1920, 1080))

        # Run parking model inference
        parking_results = parking_model(resized_frame)
        parking_detections = parking_results[0].boxes

        parked_cars = 0
        available_spaces = 0

        # Define shift values for bounding boxes
        shift_x = 100
        shift_y = 100

        # Draw bounding boxes for parking detections and update parking status
        for detection in parking_detections:
            class_id = int(detection.cls[0].item())
            label = parking_model.names[class_id]

            # Calculate the bounding box coordinates
            x_min, y_min, w, h = map(int, detection.xywh[0].tolist())
            x_max, y_max = x_min + w, y_min + h

            # Apply shift to move the bounding box left and up
            x_min -= shift_x
            y_min -= shift_y
            x_max -= shift_x
            y_max -= shift_y

            # Set bounding box color based on the label
            if label.lower() == 'occupied':
                parked_cars += 1
                color = (0, 0, 255)  # Red for occupied parking spaces
            elif label.lower() == 'empty':
                available_spaces += 1
                color = (0, 255, 0)  # Green for available parking spaces
            else:
                continue

            # Draw the bounding boxes for parking spaces
            cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(resized_frame, f"{label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Run object detection model to detect people
        object_results = object_model(resized_frame)
        object_detections = object_results[0].boxes

        for obj_detection in object_detections:
            obj_class_id = int(obj_detection.cls[0].item())
            obj_label = object_model.names[obj_class_id]

            if obj_label.lower() == 'person':
                bbox = obj_detection.xywh[0].tolist()
                x_min, y_min, x_max, y_max = int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2), int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
                
                # Apply shift
                x_min -= shift_x
                y_min -= shift_y
                x_max -= shift_x
                y_max -= shift_y

                # Draw blue bounding box for people
                cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(resized_frame, f"{obj_label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Update parking status
        parking_status['parked_cars'] = parked_cars
        parking_status['available_spaces'] = available_spaces

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            print("Error: Failed to encode frame to JPEG.")
            return None
        return jpeg.tobytes()
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

# Async generator for video streaming
async def generate_video_stream() -> AsyncGenerator[bytes, None]:
    frame_count = 0
    while not shutdown_flag.is_set():
        try:
            frame = update_parking_status()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Streamed {frame_count} frames")
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
            await asyncio.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in generate_video_stream: {e}")
            await asyncio.sleep(0.1)

# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Parking Segmentation System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .video-container { margin: 20px 0; }
            .status { background: #f0f0f0; padding: 15px; border-radius: 5px; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Parking Segmentation System</h1>
            <div class="video-container">
                <h2>Live Video Feed</h2>
                <img src="/video_feed" alt="Live Video Feed" style="max-width: 100%; border: 1px solid #ccc;">
            </div>
            <div class="status">
                <h3>Parking Status</h3>
                <p><a href="/parking_status">Get JSON Status</a></p>
                <p><a href="/docs">API Documentation</a></p>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/video_feed")
async def video_feed():
    print("Video feed requested")
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/parking_status")
async def get_parking_status():
    return JSONResponse(content=parking_status)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "camera": cap.isOpened()}

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print("\nShutting down gracefully...")
    shutdown_flag.set()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚗 Parking Segmentation System with FastAPI")
    print("📹 Video feed: http://localhost:8000/video_feed")
    print("📊 Status API: http://localhost:8000/parking_status")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🌐 Web UI: http://localhost:8000")
    
    # Run with uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        access_log=True
    ) 