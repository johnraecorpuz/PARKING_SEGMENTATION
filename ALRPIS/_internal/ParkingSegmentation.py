import cv2
from ultralytics import YOLO
import numpy as np
from flask import Flask, Response, jsonify
import threading
import signal
import sys
import os
import time

app = Flask(__name__)

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
        shift_x = 100  # Move 100 pixels to the left
        shift_y = 100  # Move 100 pixels up

        # Draw bounding boxes for parking detections and update parking status
        for detection in parking_detections:
            class_id = int(detection.cls[0].item())
            label = parking_model.names[class_id]

            # Calculate the bounding box coordinates
            x_min, y_min, w, h = map(int, detection.xywh[0].tolist())
            x_max, y_max = x_min + w, y_min + h  # Calculate the bottom-right corner

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
                continue  # Skip any other label

            # Draw the bounding boxes for parking spaces
            cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(resized_frame, f"{label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Run object detection model to detect people and draw bounding boxes
        object_results = object_model(resized_frame)
        object_detections = object_results[0].boxes

        for obj_detection in object_detections:
            obj_class_id = int(obj_detection.cls[0].item())
            obj_label = object_model.names[obj_class_id]

            if obj_label.lower() == 'person':
                bbox = obj_detection.xywh[0].tolist()
                x_min, y_min, x_max, y_max = int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2), int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
                
                # Apply shift to move the bounding box left and up for people
                x_min -= shift_x
                y_min -= shift_y
                x_max -= shift_x
                y_max -= shift_y

                # Draw blue bounding box for people
                cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue for people
                cv2.putText(resized_frame, f"{obj_label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Update parking status
        parking_status['parked_cars'] = parked_cars
        parking_status['available_spaces'] = available_spaces

        # Convert the frame to JPEG format for MJPEG streaming
        ret, jpeg = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            print("Error: Failed to encode frame to JPEG.")
            return None
        return jpeg.tobytes()
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

# Generator function to stream the camera feed
def generate_frame():
    frame_count = 0
    while not shutdown_flag.is_set():
        try:
            frame = update_parking_status()
            if frame is None:
                print("No frame available, retrying...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"Streamed {frame_count} frames")
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in generate_frame: {e}")
            time.sleep(0.1)

# Route to serve the live camera feed as MJPEG
@app.route('/video_feed')
def video_feed():
    print("Video feed requested")
    return Response(generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to send parking status as JSON
@app.route('/parking_status', methods=['GET'])
def get_parking_status():
    return jsonify(parking_status)

# Simple test route
@app.route('/')
def index():
    return """
    <html>
    <head><title>Parking Segmentation System</title></head>
    <body>
        <h1>Parking Segmentation System</h1>
        <p><a href="/video_feed">View Video Feed</a></p>
        <p><a href="/parking_status">Get Parking Status</a></p>
    </body>
    </html>
    """

# Start Flask app in a separate thread
def run_flask():
    try:
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False, debug=False)
    except Exception as e:
        print(f"Flask error: {e}")

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print("\nShutting down gracefully...")
    shutdown_flag.set()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Flask server in a separate thread (non-daemon)
    flask_thread = threading.Thread(target=run_flask, daemon=False)
    flask_thread.start()
    
    print("Parking segmentation system started. Press Ctrl+C to stop.")
    print("Flask server running on http://localhost:5000")
    print("Video feed available at http://localhost:5000/video_feed")
    print("Parking status available at http://localhost:5000/parking_status")
    
    try:
        # Keep the main thread alive
        while not shutdown_flag.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        shutdown_flag.set()
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)