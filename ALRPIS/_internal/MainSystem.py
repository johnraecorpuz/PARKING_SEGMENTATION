import cv2
import easyocr
import time
import sys
import os
import re
import numpy as np
import threading
import queue
import pymysql as mysql
from datetime import datetime
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory")

def correct_perspective(image, license_plate_corners):
    if len(license_plate_corners) != 4:
        return image
    pts1 = np.float32(license_plate_corners)
    pts2 = np.float32([[0, 0], [300, 0], [300, 100], [0, 100]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (300, 100))
    return warped

cam_indexes = {
    "face_in": 0,
    "face_out": 1,
    "plate": 2,
    "vehicle": 3
}

def resource_path(relative_path):
    """Get absolute path to resource inside PyInstaller .exe"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class PlateRecognizer:
    def __init__(self):
        # Initialize models with error handling
        try:
            self.plate_model = YOLO(resource_path('Models/bestyolov8s(4feature).pt'))
            for m in self.plate_model.model.modules():
                if hasattr(m, 'flatten_parameters'):
                    m.flatten_parameters()
            self.reader = easyocr.Reader(['en'], gpu=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
        
        # Camera control parameters
        self.zoom_levels = {
            'face_in': 1.0,
            'face_out': 1.0,
            'plate': 1.0,
            'vehicle': 1.0
        }
        self.zoom_lock = threading.Lock()
        self.max_zoom_level = 3.0
        self.zoom_step = 0.1      

        # Enhanced detection tracking system
        self.active_detections = defaultdict(dict)
        self.detection_lock = threading.RLock()
        self.min_detection_time = 3  # Minimum seconds of stable detection before logging
        self.detection_cooldown = 5  # Seconds between counting same vehicle
        
        # Thread-safe structures
        self.detected_plates = []
        self.plate_queue = queue.Queue(maxsize=100)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.gui_update_queue = queue.Queue()

        # Database configuration
        self.db_connection_retries = 3
        self.conn = self._initialize_db_connection()
        self.cursor = self.conn.cursor()

        # Vehicle tracking with enhanced state management
        self.vehicle_status = {}  # Tracks plate_number -> {status, face_name, vehicle_type, last_change}
        self.vehicle_count = 0
        self.vehicle_count_lock = threading.Lock()

        # Frame buffers with timestamps
        self.current_frames = {
            'face_in': {'frame': None, 'timestamp': 0, 'healthy': False},
            'face_out': {'frame': None, 'timestamp': 0, 'healthy': False},
            'plate': {'frame': None, 'timestamp': 0, 'healthy': False},
            'vehicle': {'frame': None, 'timestamp': 0, 'healthy': False}
        }

        # Plate validation patterns
        self.plate_patterns = [
            re.compile(r'^[0-9]{3}\s?[A-Z]{3}$'),
            re.compile(r'^[0-9]{6}$'),
            re.compile(r'^[0-9]{4}-[0-9]{7}$'),
            re.compile(r'^[0-9]{3}[A-Z]{3}$'),
            re.compile(r'^[0-9]{4}\s?[A-Z]{2}$'),
            re.compile(r'^[A-Z][0-9]{4}[A-Z]$'),
            re.compile(r'^[0-9]{2}-[A-Z]{4}$'),
            re.compile(r'^[0-9]{2}[A-Z]{4}$'),
            re.compile(r'^[A-Z]{3}\s?[0-9]{4}$'),
            re.compile(r'^[A-Z]{2}[0-9]{4}$')
        ]

       # Face recognition setup
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_database_path = r"C:\Users\Jaira Biane Maculada\Desktop\4th Year 2nd Sem\PD\face_database"
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_names = []
        self.names = []
        self.current_face_data_in = {'face_image_path': "", 'face_name': "Unknown", 'location': "Entrance"}
        self.current_face_data_out = {'face_image_path': "", 'face_name': "Unknown", 'location': "Exit"}
        self.load_face_recognizer()

        # Start background threads
        self.db_thread = threading.Thread(target=self.database_worker, daemon=True)
        self.db_thread.start()
        self.detection_processor = threading.Thread(target=self.process_detections, daemon=True)
        self.detection_processor.start()
        self.status_monitor = threading.Thread(target=self.monitor_system_status, daemon=True)
        self.status_monitor.start()

    def _initialize_db_connection(self):
        for attempt in range(self.db_connection_retries):
            try:
                conn = mysql.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="plate_recognition",
                    autocommit=True,
                    connect_timeout=5
                )
                cursor = conn.cursor()
                
                cursor.execute('''CREATE TABLE IF NOT EXISTS plate_numbers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plate_number VARCHAR(255) NOT NULL,
                    vehicle_image VARCHAR(191),
                    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    car_color VARCHAR(50),
                    location VARCHAR(100),
                    vehicle_type VARCHAR(50),
                    face_image_path VARCHAR(255),
                    face_name VARCHAR(255),
                    is_complete_record BOOLEAN DEFAULT TRUE,
                    status ENUM('IN', 'OUT') NOT NULL,
                    UNIQUE KEY (plate_number)
                )''')
                
                cursor.execute('''CREATE TABLE IF NOT EXISTS parking_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plate_number VARCHAR(255) NOT NULL,
                    vehicle_type VARCHAR(50),
                    car_color VARCHAR(50),
                    face_name VARCHAR(255),
                    status ENUM('IN', 'OUT') NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_mismatch BOOLEAN DEFAULT FALSE,
                    mismatch_reason VARCHAR(255),
                    INDEX (plate_number),
                    INDEX (timestamp)
                )''')
                
                conn.commit()
                return conn
                
            except mysql.Error as e:
                if attempt == self.db_connection_retries - 1:
                    raise RuntimeError(f"Failed to connect to database after {self.db_connection_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)

    def _has_visual_confirmation(self, detection):
        """Verify we have actual visual evidence of this detection"""
        # Check if we have recent frames showing the vehicle
        frame_keys = ['face', 'plate', 'vehicle']
        recent_frames = {
            key: self._get_latest_frame(key) 
            for key in frame_keys
        }
        
        # At least two frames should be valid and recent (within last 2 seconds)
        valid_frames = sum(
            1 for key in frame_keys 
            if (recent_frames[key] is not None and 
                recent_frames[key].size > 0 and
                time.time() - self.current_frames[key]['timestamp'] < 2)
        )
        
        return valid_frames >= 2

    def validate_detection(self, detection):
        """Validate a detection meets all requirements"""
        # Required fields
        required = {
            'plate_text': (str, lambda x: len(x) >= 3),
            'vehicle_type': (str, lambda x: x != "Unknown"),
            'car_color': (str, lambda x: x != "Unknown"),
            'face_name': (str, lambda x: True)  # Always accept face_name even if "Unknown"
        }
        
        for field, (type_check, validator) in required.items():
            if field not in detection:
                return False, f"Missing field: {field}"
            if not isinstance(detection[field], type_check):
                return False, f"Invalid type for {field}"
            if not validator(detection[field]):
                return False, f"Invalid value for {field}: {detection[field]}"
        
        return True, "Valid detection"

    def process_detections(self):
        """Background thread to process completed detections"""
        while not self.stop_event.is_set():
            current_time = time.time()
            completed = []
            
            with self.detection_lock:
                # Check all active detections
                for plate, detection in list(self.active_detections.items()):
                    # Skip if detection is too old (30 seconds)
                    if current_time - detection.get('first_detected', 0) > 30:
                        del self.active_detections[plate]
                        continue
                    
                    # Verify we have all required data for at least min_detection_time seconds
                    required_fields = ['plate_text', 'vehicle_type', 'car_color', 'face_name']
                    if (all(k in detection for k in required_fields) and 
                        (current_time - detection['first_detected'] >= self.min_detection_time)):
                        
                        # Additional check - must have been updated recently
                        if current_time - detection.get('last_updated', 0) < 2:
                            completed.append(detection)
                            del self.active_detections[plate]
            
            # Process completed detections with validation
            for detection in completed:
                is_valid, validation_msg = self.validate_detection(detection)
                if is_valid and self._has_visual_confirmation(detection):
                    self.finalize_detection(detection)
                else:
                    print(f"⚠️ Invalid detection skipped - {validation_msg}")
            
            time.sleep(1)  # Check every second

    def finalize_detection(self, detection):
        """Process a validated detection with all required data"""
        # Determine if this is a mismatch (face is "Unknown")
        is_unknown_face = detection['face_name'] == "Unknown"
        
        status, is_mismatch, reason = self.check_vehicle_movement(
            detection['plate_text'],
            detection['vehicle_type'],
            detection['face_name']
        )

         # Override mismatch status if face is unknown
        if is_unknown_face:
            is_mismatch = True
            reason = "Unknown face"
        
        # Only proceed if status is valid
        if status not in ('IN', 'OUT'):
            print(f"⚠️ Invalid Plate - not processing: {status}")
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        detection_data = {
            'time': timestamp,
            'plate_text': detection['plate_text'],
            'vehicle_type': detection['vehicle_type'],
            'car_color': detection['car_color'],
            'face_name': detection['face_name'],
            'status': status,
            'is_mismatch': is_mismatch,
            'mismatch_reason': reason
        }
        
        # Update GUI
        with self.lock:
            self.detected_plates.append(detection_data)
            if len(self.detected_plates) > 20:
                self.detected_plates.pop(0)
            
            self.gui_update_queue.put(('detection', detection_data))
            self.gui_update_queue.put(('count', self.vehicle_count))
        
        # Always save to parking records
        self.save_parking_record(
            detection['plate_text'],
            detection['vehicle_type'],
            detection['car_color'],
            detection['face_name'],
            status,
            is_mismatch,
            reason
        )
        
        # Only save to main DB if face is known and no mismatch
        if not is_unknown_face and not is_mismatch:
            self.save_to_db(
                detection['plate_text'],
                detection['car_color'],
                detection['location'],
                detection['vehicle_type'],
                detection.get('face_image_path', ""),
                detection['face_name']
            )

    def check_vehicle_movement(self, plate_text, vehicle_type, face_name):
        current_time = time.time()
        
        if plate_text not in self.vehicle_status:
            with self.vehicle_count_lock:
                self.vehicle_status[plate_text] = {
                    'status': 'IN',
                    'face_name': face_name,
                    'vehicle_type': vehicle_type,
                    'last_change': current_time
                }
                self.vehicle_count += 1
            return ('IN', False, None)
        
        last_record = self.vehicle_status[plate_text]
        
        # Check cooldown period
        if current_time - last_record['last_change'] < self.detection_cooldown:
            return (None, True, "In cooldown period")
        
        # Check for mismatches - only if status is changing from IN to OUT
        if last_record['status'] == 'IN':
            if face_name == "Unknown":
                return ('OUT', True, "Unknown face detected")
            elif last_record['face_name'] != face_name:
                return ('OUT', True, f"Driver mismatch (Previous: {last_record['face_name']}, Current: {face_name})")
            elif last_record['vehicle_type'] != vehicle_type:
                return ('OUT', True, f"Vehicle type mismatch (Previous: {last_record['vehicle_type']}, Current: {vehicle_type})")
        
        # Determine new status
        new_status = 'OUT' if last_record['status'] == 'IN' else 'IN'
        
        with self.vehicle_count_lock:
            if new_status == 'IN':
                self.vehicle_count += 1
            else:
                self.vehicle_count = max(0, self.vehicle_count - 1)
                
            self.vehicle_status[plate_text].update({
                'status': new_status,
                'face_name': face_name if new_status == 'IN' else last_record['face_name'],
                'vehicle_type': vehicle_type,
                'last_change': current_time
            })
        
        return (new_status, False, None)

    def monitor_system_status(self):
        """Background thread to monitor and log system status"""
        while not self.stop_event.is_set():
            self.print_detection_stats()
            time.sleep(10)  # Print stats every 10 seconds

    def print_detection_stats(self):
        """Print current detection statistics"""
        with self.detection_lock:
            print("\n=== Current Detection Stats ===")
            print(f"Active detections: {len(self.active_detections)}")
            print(f"Vehicle count: {self.vehicle_count}")
            for plate, det in self.active_detections.items():
                print(f"Plate: {plate}, Status: {det.get('status', 'N/A')}, " 
                      f"Age: {time.time()-det['first_detected']:.1f}s")
            print("=============================\n")

    def load_face_recognizer(self):
        face_images = []
        labels = []
        label_encoder = LabelEncoder()

        if os.path.exists(self.face_database_path):
            for folder in os.listdir(self.face_database_path):
                person_folder = os.path.join(self.face_database_path, folder)
                if os.path.isdir(person_folder):
                    for image_name in os.listdir(person_folder):
                        image_path = os.path.join(person_folder, image_name)
                        if image_name.lower().endswith((".jpg", ".png")):
                            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                face_images.append(img)
                                labels.append(folder)

            if labels:
                self.face_names = label_encoder.fit_transform(labels)
                self.names = label_encoder.classes_
                self.face_recognizer.train(face_images, np.array(self.face_names))
                print(f"Trained face recognizer with {len(self.names)} individuals")
            else:
                print("No face images found in database directory")
        else:
            print("Face database directory not found")

    def recognize_face(self, gray, face_crop):
        try:
            face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            face_crop_gray = cv2.resize(face_crop_gray, (100, 100))
            label, confidence = self.face_recognizer.predict(face_crop_gray)
            if confidence < 100 and len(self.names) > 0:
                return self.names[label]
            return "Unknown"
        except Exception as e:
            print(f"Face recognition error: {e}")
            return "Unknown"

    def extract_plate_text(self, image):
        try:
            results = self.reader.readtext(image, detail=0)
            for text in results:
                text = re.sub(r'[^A-Z0-9-]', '', text.upper())
                if self.validate_plate(text):
                    return text
            return ""
        except Exception as e:
            print(f"Plate text extraction error: {e}")
            return ""

    def validate_plate(self, plate_text):
        return any(pattern.match(plate_text) for pattern in self.plate_patterns)

    def detect_vehicle_type(self, frame, x1, y1, x2, y2):
        try:
            vehicle_crop = frame[y1:y2, x1:x2]
            results = self.plate_model(vehicle_crop)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        return self.plate_model.names[cls_id].capitalize()
            return "Unknown"
        except Exception as e:
            print(f"Vehicle type detection error: {e}")
            height, width = y2-y1, x2-x1
            area = height * width
            if area < 5000: return "Motorcycle"
            elif 5000 <= area < 15000: return "Car"
            elif 15000 <= area < 30000: return "Sedan"
            return "Large Vehicle"

    def detect_color(self, image, x1, y1, x2, y2):
        vehicle_region = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2HSV)
        
        colors = {
            "Red": [(0, 100, 100), (10, 255, 255)],
            "Gray": [(0, 0, 50), (180, 10, 200)],
            "Green": [(36, 50, 70), (89, 255, 255)],
            "Blue": [(90, 50, 70), (128, 255, 255)],
            "Yellow": [(25, 50, 70), (35, 255, 255)],
            "Black": [(0, 0, 0), (180, 255, 50)],
            "White": [(0, 0, 200), (180, 60, 255)],
            "Beige": [(20, 30, 150), (40, 80, 220)],
        }
        
        for color_name, (lower, upper) in colors.items():
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            color_percentage = np.sum(mask) / (mask.shape[0] * mask.shape[1])
            if color_percentage > 0.1:
                return color_name
        
        return "Unknown"

    def save_parking_record(self, plate_text, vehicle_type, car_color, face_name, status, is_mismatch=False, mismatch_reason=None):
        try:
            query = """INSERT INTO parking_records (
                plate_number, vehicle_type, car_color, face_name, 
                status, is_mismatch, mismatch_reason
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            
            self.cursor.execute(query, (
                str(plate_text), str(vehicle_type), str(car_color), 
                str(face_name), str(status), is_mismatch, mismatch_reason
            ))
            self.conn.commit()
            print(f"✅ Saved parking record: {plate_text} (Mismatch: {is_mismatch}, Reason: {mismatch_reason})")
            return True
        except mysql.Error as e:
            print(f"❌ Error saving parking record: {e}")
            return False

    def save_to_db(self, plate_text, car_color, location, vehicle_type, face_image_path, face_name):
        try:
            if not self.ensure_db_connection():
                print("❌ Database connection error")
                return False

            # Only require plate_text, car_color, and vehicle_type
            if not all([plate_text, car_color != "Unknown", vehicle_type != "Unknown"]):
                print(f"⚠️ Incomplete data - not saved to DB: {plate_text}")
                return False

            try:
                insert_query = """INSERT INTO plate_numbers (
                    plate_number, car_color, location, vehicle_type, 
                    face_image_path, face_name, is_complete_record
                ) VALUES (%s, %s, %s, %s, %s, %s, TRUE)"""
                
                self.cursor.execute(insert_query, (
                    str(plate_text), str(car_color), str(location), 
                    str(vehicle_type), str(face_image_path), str(face_name)
                ))
                self.conn.commit()
                print(f"✅ Saved complete record to DB: {plate_text}")
                return True
                
            except mysql.errors.IntegrityError:
                update_query = """UPDATE plate_numbers SET
                    car_color = %s, location = %s, vehicle_type = %s,
                    face_image_path = %s, face_name = %s, is_complete_record = TRUE
                    WHERE plate_number = %s"""
                
                self.cursor.execute(update_query, (
                    str(car_color), str(location), str(vehicle_type),
                    str(face_image_path), str(face_name), str(plate_text)
                ))
                self.conn.commit()
                print(f"🔄 Updated complete record in DB: {plate_text}")
                return True
                
        except mysql.Error as e:
            print(f"❌ DB Error: {e}")
            self.ensure_db_connection()
            return False

    def ensure_db_connection(self):
        try:
            if not hasattr(self, 'conn'):
                raise Exception("No database connection found.")

            try:
                # Try pinging the database; will raise if not connected
                self.conn.ping(reconnect=True)
            except:
                print("Reconnecting to database...")
                try:
                    self.conn.close()
                except:
                    pass

                self.conn = mysql.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="plate_recognition",
                    autocommit=True
                )
                self.cursor = self.conn.cursor()

            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def database_worker(self):
        while not self.stop_event.is_set():
            try:
                time.sleep(1)
            except Exception as e:
                print(f"Database worker error: {e}")

    def _get_latest_frame(self, frame_type):
        with self.lock:
            return self.current_frames[frame_type]['frame']

    def _update_frame(self, frame_type, frame):
        with self.lock:
            self.current_frames[frame_type] = {
                'frame': frame.copy() if frame is not None else None,
                'timestamp': time.time()
            }
    
    def apply_digital_zoom(self, frame, zoom_factor):
            """Apply digital zoom to frame"""
            if zoom_factor <= 1.0:
                return frame
                
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            new_w = int(w / zoom_factor)
            new_h = int(h / zoom_factor)
            
            x1 = max(0, center_x - new_w // 2)
            y1 = max(0, center_y - new_h // 2)
            x2 = min(w, center_x + new_w // 2)
            y2 = min(h, center_y + new_h // 2)
            
            zoomed = frame[y1:y2, x1:x2]
            return cv2.resize(zoomed, (w, h))

    def process_camera_face_in(self, cam_index, location, cam_name):
        cap = cv2.VideoCapture(cam_index)
        cap.set(3, 640)
        cap.set(4, 360)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (fx, fy, fw, fh) in faces:
                face_crop = frame[fy:fy + fh, fx:fx + fw]
                face_image_path = f"faces/entrance_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                cv2.imwrite(face_image_path, face_crop)
                face_name = self.recognize_face(gray, face_crop)
                
                with self.lock:
                    self.current_face_data = {
                        'face_image_path': face_image_path,
                        'face_name': face_name,
                        'location': location
                    }
                
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                cv2.putText(frame, face_name, (fx, fy - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            self._update_frame('face_in', frame)
            time.sleep(0.03)

        cap.release()
    
    def process_camera_face_out(self, cam_index, location, cam_name):
        cap = cv2.VideoCapture(cam_index)
        cap.set(3, 640)
        cap.set(4, 360)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (fx, fy, fw, fh) in faces:
                face_crop = frame[fy:fy + fh, fx:fx + fw]
                face_image_path = f"faces/exit_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                cv2.imwrite(face_image_path, face_crop)
                face_name = self.recognize_face(gray, face_crop)
                
                with self.lock:
                    self.current_face_data = {
                        'face_image_path': face_image_path,
                        'face_name': face_name,
                        'location': location
                    }
                
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                cv2.putText(frame, face_name, (fx, fy - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            self._update_frame('face_out', frame)
            time.sleep(0.03)

        cap.release()

    def process_camera_plate(self, cam_index, location, cam_name):
        cap = cv2.VideoCapture(cam_index)
        cap.set(3, 640)
        cap.set(4, 360)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.plate_model(frame, verbose=False)
            detected_plate = ""

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    if hasattr(box, 'pts') and len(box.pts) == 4:
                        plate_crop = correct_perspective(frame, box.pts)

                    plate_text = self.extract_plate_text(plate_crop)
                    if plate_text:
                        detected_plate = plate_text
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break

            self._update_frame('plate', frame)
            time.sleep(0.03)

        cap.release()

    def process_camera_vehicle(self, cam_index, location, cam_name):
        cap = cv2.VideoCapture(cam_index)
        cap.set(3, 640)
        cap.set(4, 360)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # Get current face data
            with self.lock:
                face_data = self.current_face_data.copy()

            # Plate detection
            plate_text = ""
            plate_image = self._get_latest_frame('plate')
            if plate_image is not None:
                plate_text = self.extract_plate_text(plate_image)

            if plate_text:
                # Vehicle detection
                vehicle_type = "Unknown"
                car_color = "Unknown"
                height, width = frame.shape[:2]
                vehicle_region = frame[0:int(height*0.8), 0:width]
                
                results = self.plate_model(vehicle_region)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        vehicle_type = self.plate_model.names[cls_id]
                        car_color = self.detect_color(vehicle_region, x1, y1, x2, y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(frame, f"{vehicle_type} ({car_color})", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (255, 0, 255), 2)
                        break

                # Only update if we have valid data
                if plate_text and vehicle_type != "Unknown" and car_color != "Unknown":
                    with self.detection_lock:
                        if plate_text not in self.active_detections:
                            self.active_detections[plate_text] = {
                                'first_detected': time.time(),
                                'location': location
                            }
                        
                        # Update detection data
                        self.active_detections[plate_text].update({
                            'plate_text': plate_text,
                            'vehicle_type': vehicle_type,
                            'car_color': car_color,
                            'face_name': face_data['face_name'],
                            'face_image_path': face_data['face_image_path'],
                            'last_updated': time.time()
                        })

            self._update_frame('vehicle', frame)
            time.sleep(0.03)

        cap.release()
    
    def start_processing(self):
        """Start all camera processing threads"""
        self.face_in_thread = threading.Thread(
            target=self.process_camera_face_in,
            args=(cam_indexes['face_in'], "Entrance", "Face-In Camera"),
            daemon=True
        )
        self.face_out_thread = threading.Thread(
            target=self.process_camera_face_out,
            args=(cam_indexes['face_out'], "Exit", "Face-Out Camera"),
            daemon=True
        )
        self.plate_thread = threading.Thread(
            target=self.process_camera_plate,
            args=(cam_indexes['plate'], "Entrance", "Plate Camera"),
            daemon=True
        )
        self.vehicle_thread = threading.Thread(
            target=self.process_camera_vehicle,
            args=(cam_indexes['vehicle'], "Entrance", "Vehicle Camera"),
            daemon=True
        )
        
        self.face_in_thread.start()
        # self.face_out_thread.start()
        self.plate_thread.start()
        self.vehicle_thread.start()
        
        print("All camera processing threads started")

    def get_current_detections(self):
            """Get current detection data for GUI"""
            with self.lock:
                return {
                    'detections': self.detected_plates[-10:],  # Last 10 detections
                    'vehicle_count': self.vehicle_count,
                    'current_frames': {
                        'face_in': self._get_latest_frame('face_in'),
                        'face_out': self._get_latest_frame('face_out'),
                        'plate': self._get_latest_frame('plate'),
                        'vehicle': self._get_latest_frame('vehicle')
                    },
                    'face_data': {
                        'in': self.current_face_data_in,
                        'out': self.current_face_data_out
                    }
                }

    def shutdown(self):
        self.stop_event.set()
        time.sleep(1)  # Allow threads to finish

        try:
            if hasattr(self, 'conn'):
                self.cursor.close()
                self.conn.close()
        except Exception as e:
            print(f"❌ Error closing database: {e}")

        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    recognizer = PlateRecognizer()
    try:
        recognizer.start_processing()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recognizer.stop_processing()
    except Exception as e:
        print(f"Fatal error: {e}")
        recognizer.stop_processing()