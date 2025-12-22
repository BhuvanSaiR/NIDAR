#!/usr/bin/env python3
"""
capture_visdrone_human_raspi_cam.py

- Captures frames from Raspberry Pi Camera using Picamera2
- Runs each frame through fine-tuned VisDrone YOLO model (best.pt)
- Detects ONLY humans (class 0 = person)
- Displays annotated preview
- Shows inference time and FPS
- Press 'q' to quit
"""

import time
import signal
import sys

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ---------- user parameters ----------
preview_window_name = "VisDrone YOLO - Human Detection (Raspberry Pi)"
confidence_threshold = 0.4
imgsz = 512                    # IMPORTANT: 1024 is too slow on Pi
camera_resolution = (640, 480)  # Safe for Pi

# ---------- load TRAINED model ----------
MODEL_PATH = "best.pt"  # put best.pt in same folder or give full path
model = YOLO(MODEL_PATH)

print("? Loaded model:", MODEL_PATH)
print("?? GPU not used (Raspberry Pi = CPU only)")

print("Starting Raspberry Pi camera human detection. Press 'q' to quit.")

# ---------- graceful shutdown ----------
running = True
def handle_sigint(sig, frame):
    global running
    running = False
signal.signal(signal.SIGINT, handle_sigint)

# ---------- Raspberry Pi Camera setup ----------
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": camera_resolution}
)
picam2.configure(camera_config)
picam2.start()

cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)

start_time = time.perf_counter()
frame_idx = 0

try:
    while running:
        frame_idx += 1

        # Capture RGB frame (H x W x 3)
        frame_rgb = picam2.capture_array()

        # ---------- inference ----------
        infer_start = time.perf_counter()
        results = model(
            frame_rgb,
            imgsz=imgsz,
            conf=confidence_threshold,
            classes=[0],     # ONLY PERSON
            verbose=False
        )[0]
        infer_ms = (time.perf_counter() - infer_start) * 1000

        annotated = frame_rgb.copy()

        # ---------- draw detections ----------
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"Person {conf:.2f}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ---------- FPS ----------
        elapsed = time.perf_counter() - start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0

        # ---------- overlay inference stats ----------
        cv2.putText(
            annotated,
            f"Inference: {infer_ms:.1f} ms | FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        # RGB ? BGR for OpenCV display
        cv2.imshow(
            preview_window_name,
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    total_time = time.perf_counter() - start_time
    print("\n?? Stopped")
    print(f"Frames processed: {frame_idx}")
    print(f"Average FPS: {frame_idx / total_time:.2f}")
    sys.exit(0)
