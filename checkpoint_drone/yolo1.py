#!/usr/bin/env python3
"""
yolo11_picamera2_dual_record.py
Runs YOLO11n on two Picamera2 feeds, displays annotated previews, and records each annotated feed.
"""

import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Try importing Picamera2
try:
    from picamera2 import Picamera2
except Exception as e:
    print(e)
    raise SystemExit("Cannot import Picamera2. Install it (sudo apt install python3-picamera2) and ensure libcamera is available.") from e

# --------- CONFIG ----------
CAM0_NUM = 0           # Picamera2 camera_num for camera 0
CAM1_NUM = 1           # Picamera2 camera_num for camera 1
WINDOW0 = "CAM 0"
WINDOW1 = "CAM 1"
IMG_SIZE = 640         # model input size (YOLO will resize internally)
CONF_THRESH = 0.25     # detection confidence threshold
PREVIEW_SIZE = (640, 480)  # desired capture size from camera (width, height)
SAVE_FPS = 20              # fps for saved video files
OUT0 = "cam0_output.mp4"   # output filename for cam0
OUT1 = "cam1_output.mp4"   # output filename for cam1
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
# --------------------------

# Load model (single instance)
model = YOLO("best.pt")
model_lock = threading.Lock()

def convert_picam_to_bgr(arr):
    """
    Convert Picamera2's capture_array output to OpenCV BGR.
    """
    if arr is None:
        return None
    # If grayscale-like
    if arr.ndim == 2:
        try:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception:
            return np.stack([arr]*3, axis=-1)
    # If channels present
    if arr.ndim == 3:
        h, w, ch = arr.shape
        if ch == 4:
            try:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            except Exception:
                # drop alpha, swap R<->B
                bgr = arr[..., :3][..., ::-1]
                return bgr
        if ch == 3:
            try:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                return arr[..., ::-1]
    # fallback: return as-is (may be accepted by cv2)
    return arr

def annotate_frame(frame, results, names, conf_thresh=0.25):
    """
    Draw boxes & labels onto frame using ultralytics Results object.
    Returns annotated frame (copy).
    """
    if frame is None:
        return None
    annotated = frame.copy()
    if results is None:
        return annotated

    # results is usually list-like; pick first element
    r = results[0] if hasattr(results, '__len__') else results

    boxes = getattr(r, "boxes", None)
    # If no boxes attribute or zero detections
    try:
        if boxes is None or len(boxes) == 0:
            return annotated
    except Exception:
        # Some versions don't support len(); continue and handle empties later
        pass

    # try to extract arrays (supports torch tensors -> cpu().numpy())
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        try:
            xyxy = np.array(boxes.xyxy)
            confs = np.array(boxes.conf)
            cls_ids = np.array(boxes.cls).astype(int)
        except Exception:
            return annotated

    for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = names.get(int(cid), str(int(cid)))
        txt = f"{label} {float(conf):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0,200,0), -1)
        cv2.putText(annotated, txt, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return annotated

class PicamWorker(threading.Thread):
    def __init__(self, cam_num, window_name, model, model_lock, preview_size=(640,480), imgsz=640, conf_thresh=0.25, out_file=None, save_fps=20):
        super().__init__(daemon=True)
        self.cam_num = cam_num
        self.window_name = window_name
        self.model = model
        self.model_lock = model_lock
        self.preview_size = preview_size
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.running = True
        self.picam = None
        self.out_file = out_file
        self.save_fps = save_fps
        self.writer = None
        self.frame_shape = None

    def setup_camera(self):
        # Construct Picamera2 for a specific camera number
        try:
            self.picam = Picamera2(camera_num=self.cam_num)
        except TypeError:
            try:
                self.picam = Picamera2(self.cam_num)
            except Exception:
                self.picam = Picamera2()

        # Configure preview
        try:
            cfg = self.picam.create_preview_configuration({"size": self.preview_size})
            self.picam.configure(cfg)
        except Exception:
            # fallback: let Picamera2 choose defaults
            pass
        self.picam.start()

    def ensure_writer(self, frame):
        """
        Initialize VideoWriter when we know frame size.
        """
        if self.out_file is None:
            return
        if self.writer is None:
            h, w = frame.shape[:2]
            self.frame_shape = (w, h)
            self.writer = cv2.VideoWriter(self.out_file, FOURCC, float(self.save_fps), self.frame_shape)
            if not self.writer.isOpened():
                print(f"[{self.window_name}] WARNING: VideoWriter failed to open file {self.out_file}")

    def run(self):
        try:
            self.setup_camera()
        except Exception as e:
            print(f"[{self.window_name}] Failed to start Picamera2 (cam_num={self.cam_num}): {e}")
            return

        fps_time = time.time()
        frame_count = 0
        while self.running:
            try:
                arr = self.picam.capture_array()
            except Exception as e:
                print(f"[{self.window_name}] capture_array error: {e}")
                time.sleep(0.05)
                continue

            frame_bgr = convert_picam_to_bgr(arr)
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            # Inference (serialized model)
            with self.model_lock:
                results = self.model(frame_bgr, imgsz=self.imgsz)

            annotated = annotate_frame(frame_bgr, results, self.model.names, conf_thresh=self.conf_thresh)
            if annotated is None:
                annotated = frame_bgr

            # init writer when first frame arrives
            self.ensure_writer(annotated)

            # write annotated frame to file if writer ready
            if self.writer is not None and self.writer.isOpened():
                try:
                    self.writer.write(annotated)
                except Exception as e:
                    print(f"[{self.window_name}] writer.write error: {e}")

            # show FPS occasionally
            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                fps = 10 / (now - fps_time + 1e-6)
                fps_time = now
                cv2.putText(annotated, f"{fps:.1f} FPS", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow(self.window_name, annotated)

            # global quit: pressing 'q' in any window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        # cleanup
        try:
            self.picam.stop()
        except Exception:
            pass
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        cv2.destroyWindow(self.window_name)

    def stop(self):
        self.running = False

def main():
    print("Starting YOLO11n + Picamera2 dual-camera demo. Press 'q' in any window to quit.")
    w0 = PicamWorker(CAM0_NUM, WINDOW0, model, model_lock, preview_size=PREVIEW_SIZE,
                     imgsz=IMG_SIZE, conf_thresh=CONF_THRESH, out_file=OUT0, save_fps=SAVE_FPS)
    w1 = PicamWorker(CAM1_NUM, WINDOW1, model, model_lock, preview_size=PREVIEW_SIZE,
                     imgsz=IMG_SIZE, conf_thresh=CONF_THRESH, out_file=OUT1, save_fps=SAVE_FPS)
    w0.start()
    time.sleep(0.1)
    w1.start()

    try:
        while (w0.is_alive() or w1.is_alive()):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        w0.stop()
        w1.stop()
        w0.join(timeout=2.0)
        w1.join(timeout=2.0)
        # ensure any remaining writers closed
        if hasattr(w0, "writer") and w0.writer is not None:
            try: w0.writer.release()
            except: pass
        if hasattr(w1, "writer") and w1.writer is not None:
            try: w1.writer.release()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
