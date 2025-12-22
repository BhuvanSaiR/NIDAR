# app.py
# Flask + OpenCV MJPEG streaming with a background capture thread.
# Works on Raspberry Pi (Linux) with a USB UVC webcam (e.g. Zebronics).

from flask import Flask, Response, render_template, send_file
import cv2
import threading
import time

app = Flask(__name__)

class Camera:
    def __init__(self, device=0, width=1280, height=720, fps=20, api_preference=cv2.CAP_V4L2):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.api_pref = api_preference

        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self._open_camera()

        # Start capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _open_camera(self):
        # Try to (re)open the camera
        cap = cv2.VideoCapture(self.device, self.api_pref)
        if not cap.isOpened():
            # fallback without api hint
            cap = cv2.VideoCapture(self.device)
        # try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap = cap

    def _capture_loop(self):
        # Continuously capture frames. If camera fails, attempt to reopen.
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                try:
                    self._open_camera()
                except Exception as e:
                    print("Camera open error:", e)
                    time.sleep(2)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                # camera read failed; reopen
                print("Warning: frame grab failed, reopening camera...")
                try:
                    self.cap.release()
                except:
                    pass
                time.sleep(1)
                self._open_camera()
                continue

            # Optionally: resize or rotate here if needed:
            # frame = cv2.resize(frame, (self.width, self.height))
            # frame = cv2.rotate(frame, cv2.ROTATE_180)  # if your camera is upside down

            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            with self.lock:
                self.frame = jpeg.tobytes()

            # sleep depending on fps
            if self.fps > 0:
                time.sleep(1.0 / self.fps)

    def get_frame(self):
        with self.lock:
            return self.frame

    def snapshot(self, path="/tmp/snapshot.jpg"):
        data = self.get_frame()
        if data:
            with open(path, 'wb') as f:
                f.write(data)
            return path
        return None

    def stop(self):
        self.stopped = True
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except:
            pass
        self.thread.join(timeout=1)


# Configure camera: device=0 corresponds to /dev/video0; if your webcam is different use e.g. '/dev/video1' or 1
camera = Camera(device=0, width=1280, height=720, fps=20)

@app.route('/')
def index():
    return render_template('index.html')

def generate_mjpeg():
    boundary = b'--frame'
    while True:
        frame = camera.get_frame()
        if not frame:
            # No frame yet; yield a tiny pause so client doesn't time out
            time.sleep(0.1)
            continue
        yield boundary + b'\r\n' + b'Content-Type: image/jpeg\r\n' + b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n\r\n' + frame + b'\r\n'

@app.route('/video_feed')
def video_feed():
    # Flask response for MJPEG stream.
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    path = camera.snapshot()
    if path:
        return send_file(path, mimetype='image/jpeg')
    return ("No snapshot available", 503)

if __name__ == '__main__':
    # For development/testing:
    # app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    # For a more production-ish test, use gunicorn (see instructions).
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
