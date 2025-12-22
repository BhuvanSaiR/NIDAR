#!/usr/bin/env python3
"""
pi5_yolo11n_capture.py

- Captures frames from Raspberry Pi camera (Picamera2) at ~20 FPS (every 50 ms)
- Forces exposure time to 1 ms (1000 microseconds) and disables auto-exposure
- Runs YOLO11n inference (Ultralytics) and overlays boxes + inference time on preview
- Designed for Raspberry Pi 5; recommended to use ONNX export for faster CPU inference.
"""

import time
import threading
import queue
import numpy as np
import cv2

# Picamera2 (libcamera wrapper)
from picamera2 import Picamera2, Preview

# Choose your inference backend:
# Option A: Ultralytics (simple & robust; can accept .pt or exported .onnx with the Ultralytics runtime)
#   pip install ultralytics
# Option B: onnxruntime (fast CPU inference if you exported to ONNX)
#   pip install onnxruntime
use_ultralytics = False

if use_ultralytics:
    from ultralytics import YOLO
else:
    import onnxruntime as ort

# --------- Config ----------
MODEL_PATH = "best.pt"    # change to "yolo11n.onnx" if using onnx runtime
CAM_RES = (640, 640)         # keep small for speed; YOLO11n typical input 640
TARGET_DT = 0.050            # seconds between captures -> 50 ms
EXPOSURE_US = 1000           # 1 ms = 1000 microseconds
MAX_QUEUE = 3                # buffer size; drops frames when overloaded
CONF_THRESH = 0.25
# --------------------------

# Frame queue: camera -> inference
frame_q = queue.Queue(maxsize=MAX_QUEUE)
stop_event = threading.Event()

def camera_thread():
    """
    Capture loop: enqueues frames at ~TARGET_DT intervals. Uses manual exposure.
    """
    picam2 = Picamera2()
    # preview config: match CAM_RES
    preview_config = picam2.create_preview_configuration(main={"size": CAM_RES})
    picam2.configure(preview_config)

    # Set manual exposure (ExposureTime in microseconds), disable AE so it's honored
    # AnalogueGain may be set to 1.0 as starting point
    picam2.set_controls({"ExposureTime": EXPOSURE_US, "AnalogueGain": 1.0, "AeEnable": False})

    # Start preview window (will open a window on the Pi's desktop)
    try:
        picam2.start_preview(Preview.QTGL)
    except Exception:
        # if no desktop, still proceed without preview call
        pass

    picam2.start()
    print("[camera] started, exposure set to", EXPOSURE_US, "µs. Target interval:", TARGET_DT, "s")

    next_ts = time.time()
    try:
        while not stop_event.is_set():
            t0 = time.time()
            # capture_array() returns an RGB numpy array (H, W, 3) (uint8)
            frame = picam2.capture_array()  # blocking call for one frame
            # convert to BGR for OpenCV drawing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # push into queue (drop oldest if full to keep recent)
            try:
                frame_q.put_nowait((time.time(), frame_bgr))
            except queue.Full:
                try:
                    _ = frame_q.get_nowait()  # drop oldest
                    frame_q.put_nowait((time.time(), frame_bgr))
                except queue.Empty:
                    pass

            # wait to maintain target DT (but if capture blocked longer, we skip sleeping)
            next_ts += TARGET_DT
            sleep_for = next_ts - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # we're behind; adjust next_ts to now to avoid drift
                next_ts = time.time()
    finally:
        picam2.stop()
        picam2.close()
        print("[camera] stopped")

def draw_boxes(frame, boxes, confs, classes, class_names=None):
    """Draw boxes onto frame (BGR), boxes in xyxy numpy (N,4)"""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = float(confs[i])
        cls = int(classes[i])
        label = f"{cls}:{conf:.2f}" if class_names is None else f"{class_names[cls]}:{conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, label, (x1, max(10, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

def infer_thread():
    """
    Pulls frames from frame_q, runs inference, and displays a preview with overlayed boxes + inference time.
    Uses:
      - Ultralytics YOLO API if use_ultralytics==True
      - else ONNX Runtime (basic) if MODEL_PATH is .onnx
    """
    # Load model
    if use_ultralytics:
        print("[inference] loading Ultralytics model:", MODEL_PATH)
        model = YOLO(MODEL_PATH)  # Ultralytics will use best backend available (if exported ONNX it may use a faster engine)
        # Set model to predict small images for speed; override imgsz per-predict below
    else:
        print("[inference] loading ONNX model:", MODEL_PATH)
        sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name

    cv2.namedWindow("YOLO11n (Pi5)", cv2.WINDOW_NORMAL)
    last_display = time.time()
    try:
        while not stop_event.is_set():
            try:
                ts, frame = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Preprocessing: model expects RGB; Ultralytics accepts BGR as well often, but we'll send RGB to be consistent
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t_in0 = time.time()
            if use_ultralytics:
                # Ultralytics: pass numpy array directly; set imgsz to CAM_RES[0] (assumes square)
                # model.predict returns a Results object; we'll access boxes via results[0].boxes
                results = model.predict(source=img_rgb, imgsz=CAM_RES[0], conf=CONF_THRESH, device='cpu', verbose=False)
                infer_time = time.time() - t_in0

                # handle results (could be list-like)
                try:
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.asarray(r.boxes.xyxy)
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else np.asarray(r.boxes.conf)
                    classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else np.asarray(r.boxes.cls)
                except Exception:
                    # fallback: no detections
                    boxes = np.zeros((0,4))
                    confs = np.array([])
                    classes = np.array([])
            else:
                # ONNX path (very minimal; assumes exported model outputs standard YOLO format).
                # You should replace this with proper ONNX output postprocessing matching your exported model.
                img = cv2.resize(img_rgb, (CAM_RES[0], CAM_RES[1]))
                img_f = img.astype(np.float32) / 255.0
                img_f = np.transpose(img_f, (2,0,1))[None, ...]  # (1,C,H,W)
                outputs = sess.run(None, {input_name: img_f})
                infer_time = time.time() - t_in0

                # TODO: apply decoding + NMS according to your exported ONNX output format
                boxes = np.zeros((0,4))
                confs = np.array([])
                classes = np.array([])

            # draw boxes on the BGR frame
            draw_im = frame.copy()
            if boxes.shape[0] > 0:
                draw_boxes(draw_im, boxes, confs, classes)

            # overlay inference time and queue size
            text = f"Infer: {infer_time*1000:.1f} ms   Queue: {frame_q.qsize()}"
            cv2.putText(draw_im, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            cv2.imshow("YOLO11n (Pi5)", draw_im)
            # short waitKey to allow window refresh; also captures 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    finally:
        cv2.destroyAllWindows()
        print("[inference] stopped")

def main():
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_inf = threading.Thread(target=infer_thread, daemon=True)

    t_cam.start()
    t_inf.start()

    try:
        while t_cam.is_alive() and t_inf.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupted, stopping...")

    t_cam.join()
    t_inf.join()
    print("Exiting.")

if __name__ == "__main__":
    main()
