import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import numpy as np
import threading
import requests
import queue
import time
import cv2
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import subprocess
import os
import sys
import atexit

from MainSystem import PlateRecognizer

def resource_path(relative_path):
    """Get absolute path to resource inside PyInstaller .exe"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class ALPR_GUI:
    def __init__(self, root):
        flask_path = resource_path('ParkingSegmentation.py')
        self.flask_process = subprocess.Popen(
            [sys.executable, flask_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        self.root = root
        self.root.title("ALPRIS")
        self.root.geometry("1366x900")
        self.root.minsize(1024, 768)
        self.root.iconbitmap(resource_path("ALPRIS.ico"))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Core system state
        self.update_interval = 100
        self.stop_parking_feed = True
        self.is_running = False
        self.after_id = None
        self.vehicle_count = 0
        self.parking_spaces = 0
        self.available_spaces = 0
        self.max_log_entries = 200

        # Camera display states
        self.camera_width = 320
        self.camera_height = 240
        self.zoom_states = {
            'face_in': 1.0,
            'plate_in': 1.0,
            'vehicle_in': 1.0,
            'parking_in': 1.0
        }

        # Initialize
        self.setup_ui()
        self.init_recognition_system()
        self.schedule_gui_update()

    def setup_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header with title and buttons
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            header_frame,
            text="ALPRIS",
            font=('Helvetica', 16, 'bold')
        ).pack(side=tk.LEFT)

        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side=tk.RIGHT)

        self.start_btn = ttk.Button(
            btn_frame, text="Start", width=12,
            command=self.start_system, state=tk.NORMAL
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            btn_frame, text="Stop", width=12,
            command=self.stop_system, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Main content layout: top (parking + log), bottom (3 cameras)
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(content_frame)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(content_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self._setup_camera_section(top_frame, bottom_frame)

        self.notification_var = tk.StringVar(value="System ready")
        self.notification_label = ttk.Label(
            header_frame,
            textvariable=self.notification_var,
            font=('Helvetica', 10),
            bootstyle=INFO,
            padding=5
        )
        self.notification_label.pack(side=tk.LEFT, padx=10)

    def _setup_camera_section(self, top_frame, bottom_frame):
        self.camera_canvases = {}

        def create_camera_frame(parent, label, key):
            frame = ttk.Labelframe(parent, text=label, padding=10)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            canvas = tk.Canvas(frame, width=self.camera_width, height=self.camera_height, bg='black')
            canvas.pack(fill=tk.BOTH, expand=True)
            self.camera_canvases[key] = canvas
            self._add_zoom_controls(frame, key)

        # --- TOP HALF ---

        top_row = ttk.Frame(top_frame)
        top_row.pack(fill=tk.BOTH, expand=True)

        # Parking segmentation camera
        parking_frame = ttk.Labelframe(top_row, text="PARKING SEGMENTATION", padding=10)
        parking_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        parking_canvas_container = ttk.Frame(parking_frame, width=576, height=324)
        parking_canvas_container.pack_propagate(False)
        parking_canvas_container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(
            parking_canvas_container,
            bg='black',
            highlightthickness=0
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        self.camera_canvases['parking_in'] = canvas
        self._add_zoom_controls(parking_frame, 'parking_in')

        info_frame = ttk.Frame(parking_frame)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        ttk.Label(info_frame, text="Parked:").pack(side=tk.LEFT, padx=(10, 0))
        self.parked_label = ttk.Label(info_frame, text="0", width=4)
        self.parked_label.pack(side=tk.LEFT)

        ttk.Label(info_frame, text="Available:").pack(side=tk.LEFT, padx=(10, 0))
        self.available_label = ttk.Label(info_frame, text="0", width=4)
        self.available_label.pack(side=tk.LEFT)

        # Detection Log
        log_frame = ttk.Labelframe(top_row, text="DETECTION LOG", padding=10)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        columns = [('time', 'Time', 120), ('plate', 'Plate', 100), ('vehicle', 'Vehicle', 90),
                ('color', 'Color', 80), ('driver', 'Driver', 100),
                ('status', 'Status', 70), ('alert', 'Alert', 100)]

        self.log_tree = ttk.Treeview(
            log_frame,
            columns=[col[0] for col in columns],
            show='headings',
            selectmode='browse'
        )

        for col_id, col_text, col_width in columns:
            self.log_tree.heading(col_id, text=col_text)
            self.log_tree.column(col_id, width=col_width, anchor=tk.CENTER)

        self.log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        self.log_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._setup_log_context_menu()

        # --- BOTTOM HALF ---

        bottom_row = ttk.Frame(bottom_frame)
        bottom_row.pack(fill=tk.BOTH, expand=True)

        create_camera_frame(bottom_row, "DRIVER CAMERA", 'face_in')
        create_camera_frame(bottom_row, "LICENSE PLATE CAMERA", 'plate_in')
        create_camera_frame(bottom_row, "VEHICLE CAMERA", 'vehicle_in')

    def _add_zoom_controls(self, parent_frame, camera_key):
        control_frame = ttk.Frame(parent_frame)
        control_frame.pack(fill=tk.X, pady=(5, 0))

        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(side=tk.RIGHT)

        # Zoom out button
        ttk.Button(
            zoom_frame,
            text="-",
            command=lambda: self.adjust_zoom(camera_key, 0.9),
            bootstyle=SECONDARY,
            width=3
        ).pack(side=tk.LEFT, padx=2)

        # Zoom in button
        ttk.Button(
            zoom_frame,
            text="+",
            command=lambda: self.adjust_zoom(camera_key, 1.1),
            bootstyle=SECONDARY,
            width=3
        ).pack(side=tk.LEFT, padx=2)

        # Reset zoom
        ttk.Button(
            zoom_frame,
            text="Reset",
            command=lambda: self.reset_zoom(camera_key),
            bootstyle=SECONDARY,
            width=6
        ).pack(side=tk.LEFT, padx=2)

    def adjust_zoom(self, camera_key, factor):
        self.zoom_states[camera_key] *= factor
        self.zoom_states[camera_key] = max(0.5, min(self.zoom_states[camera_key], 3.0))

        if camera_key == 'parking_in':
            self.refresh_parking_frame()
        else:
            frame = self.plate_recognizer._get_latest_frame(camera_key.split('_')[0])
            if frame is not None:
                self.display_frame(frame, camera_key)
    
    def refresh_parking_frame(self):
        self.stop_parking_feed = False

    def reset_zoom(self, camera_key):
        self.zoom_states[camera_key] = 1.0

    def _setup_log_context_menu(self):
        self.log_menu = tk.Menu(self.root, tearoff=0)
        self.log_menu.add_command(label="View Details", command=self._view_log_details)
        self.log_menu.add_command(label="Clear Log", command=self.clear_log)
        self.log_tree.bind("<Button-3>", self._show_log_context_menu)

    def _show_log_context_menu(self, event):
        item = self.log_tree.identify_row(event.y)
        if item:
            self.log_tree.selection_set(item)
            self.log_menu.post(event.x_root, event.y_root)

    def _view_log_details(self):
        selected = self.log_tree.selection()
        if selected:
            values = self.log_tree.item(selected, 'values')
            details = "\n".join(f"{self.log_tree.heading(col)['text']}: {val}" 
                                for col, val in zip(self.log_tree['columns'], values))
            messagebox.showinfo("Detection Details", details)

    def clear_log(self):
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)

    def init_recognition_system(self):
        self.plate_recognizer = PlateRecognizer()

    def start_system(self):
        if self.is_running:
            return
        try:
            if not self.wait_for_flask_ready():
                self.show_notification("Flask server is not ready.", "alert")
                return
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

            self.plate_recognizer.start_processing()

            self.stop_parking_feed = False
            threading.Thread(target=self.update_parking_camera_feed, daemon=True).start()

            self.update_parking_info_from_flask()

            self.show_notification("System started successfully", "success")
        except Exception as e:
            self.is_running = False
            messagebox.showerror("Error", f"Failed to start system: {str(e)}")

    def stop_system(self):
        if not self.is_running:
            return
        try:
            self.is_running = False
            self.stop_btn.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.NORMAL)

            self.plate_recognizer.shutdown()
            self.stop_parking_feed = True

            if hasattr(self, 'parking_after_id'):
                self.root.after_cancel(self.parking_after_id)
                del self.parking_after_id

            canvas = self.camera_canvases.get('parking_in')
            if canvas:
                canvas.delete("all")

            self.vehicle_count = 0
            self.available_spaces = 0
            self.parking_spaces = 0
            self.parked_label.config(text="0")
            self.available_label.config(text="0", bootstyle='default')

            self.show_notification("System stopped", "info")

            self.plate_recognizer = PlateRecognizer()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop system: {str(e)}")

    def schedule_gui_update(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.update_gui()
        self.after_id = self.root.after(self.update_interval, self.schedule_gui_update)

    def update_gui(self):
        while True:
            try:
                update = self.plate_recognizer.gui_update_queue.get_nowait()
                self.process_update(update)
            except queue.Empty:
                break

        self.update_camera_displays()

    def update_parking_camera_feed(self):
        try:
            stream = requests.get("http://localhost:5000/video_feed", stream=True)
            bytes_data = b""
            for chunk in stream.iter_content(chunk_size=1024):
                if self.stop_parking_feed:
                    break
                bytes_data += chunk
                a, b = bytes_data.find(b'\xff\xd8'), bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    img_array = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.root.after(0, self.display_parking_frame, frame)
        except Exception as e:
            print(f"Parking video feed error: {e}")

    def wait_for_flask_ready(self, max_retries=10, delay=1.5):
        for attempt in range(max_retries):
            try:
                r = requests.get("http://localhost:5000/parking_status", timeout=2)
                if r.status_code == 200:
                    print("Flask server is ready")
                    return True
            except requests.exceptions.RequestException as e:
                print(f"Waiting for Flask server... ({attempt+1}/{max_retries})")
                time.sleep(delay)
        print("Flask server not available after multiple retries.")
        return False
    
    def display_parking_frame(self, frame):
        try:
            zoom_factor = self.zoom_states.get('parking_in', 1.0)

            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2

            if zoom_factor != 1.0:
                new_w = int(w / zoom_factor)
                new_h = int(h / zoom_factor)
                x1 = max(0, center_x - new_w // 2)
                y1 = max(0, center_y - new_h // 2)
                x2 = min(w, center_x + new_w // 2)
                y2 = min(h, center_y + new_h // 2)
                if x2 > x1 and y2 > y1:
                    frame = frame[y1:y2, x1:x2]

            canvas = self.camera_canvases['parking_in']
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()

            # Maintain aspect ratio
            frame_aspect = w / h
            canvas_aspect = canvas_w / canvas_h

            if frame_aspect > canvas_aspect:
                new_w = canvas_w
                new_h = int(canvas_w / frame_aspect)
            else:
                new_h = canvas_h
                new_w = int(canvas_h * frame_aspect)

            frame_resized = cv2.resize(frame, (new_w, new_h))

            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)

            canvas.imgtk = imgtk
            canvas.delete("all")
            canvas.create_image(
                canvas_w // 2,
                canvas_h // 2,
                anchor=tk.CENTER,
                image=imgtk
            )

        except Exception as e:
            print(f"Display error: {e}")

    def update_parking_info_from_flask(self):
        if not self.is_running:
            return

        try:
            res = requests.get("http://localhost:5000/parking_status", timeout=2)
            if res.status_code == 200:
                data = res.json()
                self.vehicle_count = data.get('parked_cars', 0)
                self.available_spaces = data.get('available_spaces', 0)

                self.parked_label.config(text=str(self.vehicle_count))
                self.available_label.config(text=str(self.available_spaces))

                style = 'danger' if self.available_spaces < 5 else 'success'
                self.available_label.config(bootstyle=style)
        except Exception as e:
            print(f"Error fetching parking data: {e}")

        self.parking_after_id = self.root.after(3000, self.update_parking_info_from_flask)

    def process_update(self, update):
        update_type, data = update
        if update_type == 'detection':
            self.process_detection(data)
        elif update_type == 'parking_update':
            self.process_parking_update(data)
        elif update_type == 'error':
            self.show_notification(data, "alert")

    from datetime import datetime  # ensure this is imported at the top

    def process_detection(self, data):
        if data['status'] == 'IN':
            self.vehicle_count += 1
        else:
            self.vehicle_count = max(0, self.vehicle_count - 1)

        # Determine tags
        if data['is_mismatch']:
            tags = ['mismatch']
        elif data['face_name'] == 'Unknown':
            tags = ['unknown']

            # Show notification in label
            self.show_notification("⚠️ UNKNOWN DRIVER DETECTED ⚠️", "alert")

            # Popup alert window
            plate = data.get('plate_text', 'N/A')
            vehicle = data.get('vehicle_type', 'N/A')
            timestamp = data.get('time') or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            message = (
                "⚠︎ UNKNOWN DRIVER DETECTED ⚠︎\n\n"
                "An unknown driver was detected!\n\n"
                f"Vehicle Plate: {plate}\n"
                f"Vehicle Type: {vehicle}\n"
                f"Time: {timestamp}\n\n"
                "Security personnel should verify identification."
            )
            messagebox.showwarning("Unknown Driver Alert", message)

        else:
            tags = ['normal']

        # Insert into detection log
        self.log_tree.insert('', tk.END, values=(
            data['time'],
            data['plate_text'],
            data['vehicle_type'],
            data['car_color'],
            data['face_name'],
            data['status'],
            "ALERT" if data['is_mismatch'] else ""
        ), tags=tuple(tags))

        # Auto-scroll
        self.log_tree.see(self.log_tree.get_children()[-1])

        # Trim old entries
        if len(self.log_tree.get_children()) > self.max_log_entries:
            self.log_tree.delete(self.log_tree.get_children()[0])

    def process_parking_update(self, data):
        self.parking_spaces = data['total_spaces']
        self.available_spaces = self.parking_spaces - data['occupied_spaces']
        self.vehicle_count = data['occupied_spaces']
        self.parked_label.config(text=str(self.vehicle_count))
        self.available_label.config(text=str(self.available_spaces))
        style = DANGER if self.available_spaces < 5 else SUCCESS
        self.available_label.config(bootstyle=style)

    def update_camera_displays(self):
        for key in ['face_in', 'plate', 'vehicle']:
            frame = self.plate_recognizer._get_latest_frame(key)
            if frame is not None:
                self.display_frame(frame, key + '_in' if key != 'face_in' else key)

    def display_frame(self, frame, canvas_key):
        if frame is None:
            return

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply zoom
            zoom = self.zoom_states.get(canvas_key, 1.0)
            h, w = frame.shape[:2]
            if zoom != 1.0:
                cx, cy = w // 2, h // 2
                nw, nh = int(w / zoom), int(h / zoom)
                x1, y1 = max(0, cx - nw // 2), max(0, cy - nh // 2)
                x2, y2 = min(w, cx + nw // 2), min(h, cy + nh // 2)
                if x2 > x1 and y2 > y1:
                    frame = frame[y1:y2, x1:x2]

            canvas = self.camera_canvases[canvas_key]
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()

            # Maintain aspect ratio
            frame_aspect = w / h
            canvas_aspect = canvas_w / canvas_h

            if frame_aspect > canvas_aspect:
                new_w = canvas_w
                new_h = int(canvas_w / frame_aspect)
            else:
                new_h = canvas_h
                new_w = int(canvas_h * frame_aspect)

            frame_resized = cv2.resize(frame, (new_w, new_h))

            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)

            canvas.imgtk = imgtk
            canvas.delete("all")
            canvas.create_image(
                canvas_w // 2,
                canvas_h // 2,
                anchor=tk.CENTER,
                image=imgtk
            )

        except Exception as e:
            print(f"Error displaying {canvas_key}: {e}")

    def show_notification(self, message, msg_type):
        styles = {
            "alert": (DANGER, INVERSE),
            "success": (SUCCESS, INVERSE),
            "info": (INFO, INVERSE)
        }
        style = styles.get(msg_type, (INFO, INVERSE))
        self.notification_var.set(message)
        self.notification_label.config(bootstyle=style)

    def on_closing(self):
        if self.is_running:
            if not messagebox.askokcancel("Quit", "Stop system and quit?", parent=self.root):
                return
            self.stop_system()
        if self.after_id:
            self.root.after_cancel(self.after_id)
        if hasattr(self, 'flask_process'):
            self.flask_process.terminate()
        self.root.destroy()

LOCK_FILE = os.path.join(os.getenv("TEMP"), "alpris.lock")

def already_running():
    if os.path.exists(LOCK_FILE):
        return True
    try:
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        return False
    except Exception:
        return True

def cleanup_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

if __name__ == "__main__":
    if already_running():
        print("ALPRIS is already running.")
        sys.exit(0)
    
    atexit.register(cleanup_lock)

    try:
        root = ttk.Window(
            themename="darkly",
            title="ALPRIS System",
            size=(1366, 900),
            resizable=(True, True)
        )
        app = ALPR_GUI(root)
        root.mainloop()
    except Exception as e:
        print("Fatal error:", str(e))
        import traceback
        traceback.print_exc()
        try:
            root.destroy()
        except:
            pass