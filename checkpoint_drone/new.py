#!/usr/bin/env python3
# yolo11n_ncnn_picam.py
import time
import cv2
from ultralytics import YOLO

# Picamera2 import (Raspberry Pi OS Bookworm usually has this)
try:
    from picamera2 import Picamera2
except Exception as e:
    raise SystemExit("Picamera2 not found. Install/use Raspberry Pi OS (Bookworm) or use alternate camera input.") from e

# Path to exported NCNN model directory (created by model.export(format="ncnn"))
NCNN_MODEL_DIR = "./yolo11n_ncnn_model"  # adjust if needed

# Load NCNN model (Ultralytics supports passing the exported folder)
print("[INFO] Loading NCNN model from:", NCNN_MODEL_DIR)
model = YOLO(NCNN_MODEL_DIR)  # this will use the NCNN backend under the hood

# Initialize Picamera2
picam2 = Picamera2()
# choose a resolution; keep it small for speed (e.g., 640x640 or 1280x720)
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"  # matches Ultralytics expectation in examples
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(0.5)  # let camera warm up

print("[INFO] Starting capture loop. Press 'q' in the window to quit.")
try:
    while True:
        frame = picam2.capture_array()  # numpy array, RGB
        # Run inference (frame can be numpy array)
        # returns a Results object; call model(frame) is a shorthand to predict on the image
        results = model(frame)  

        # Annotated image (Ultralytics results[0].plot() returns a plotted image)
        annotated = results[0].plot()  

        # If annotated is RGB, convert to BGR for OpenCV display
        # Ultralytics' plot often returns an OpenCV-ready image; if colors look off, uncomment conversion:
        # annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        cv2.imshow("YOLO11n NCNN - Picamera2", annotated)

        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("[INFO] Exiting.")
