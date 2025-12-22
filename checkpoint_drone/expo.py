# dual_yolo11_with_mavsdk_forward_and_servo_drop.py
# Requirements: python3, picamera2, opencv-python, ultralytics, numpy, mavsdk (optional),
#               gpiozero (for servo) if running on Raspberry Pi with a real servo.
#
# NOTE: Test in simulation / with props removed. Using a real vehicle is dangerous.

import time
import cv2
import numpy as np
import os
import asyncio
import threading
from datetime import datetime
from picamera2 import Picamera2
from ultralytics import YOLO

# Try import mavsdk; provide friendly error if not installed
try:
    from mavsdk import System
    from mavsdk.offboard import VelocityBodyYawspeed, PositionNedYaw
    MAVSDK_AVAILABLE = True
except Exception as e:
    print("[WARN] mavsdk import failed:", e)
    MAVSDK_AVAILABLE = False

# Try import gpiozero Servo (optional)
try:
    from gpiozero import Servo
    SERVO_AVAILABLE = True
except Exception as e:
    print("[WARN] gpiozero import failed or not available:", e)
    SERVO_AVAILABLE = False

# ---- Config ----
CAM_SIZE = (640, 480)
INFER_SIZE = 320
MODEL_PATH = "yolo11n.pt"
DISPLAY_WINDOW = "Dual YOLO11n"
FPS_AVG_ALPHA = 0.9
PERSON_CONF_THR = 0.3
PERSON_AREA_THRESHOLD = 0.5  # 50% area threshold

# MAV settings
SYSTEM_ADDRESS = "serial:///dev/ttyACM0:115200"  # adjust as needed
TAKEOFF_ALT = -2.0  # NED down negative = up 2m

# Servo/drop settings
SERVO_PIN = 18                 # as you specified: Servo(18)
SERVO_INIT_MAX = True          # call servo.max() at startup if True
DROP_HOLD_SECONDS = 0.8        # how long to hold servo.min() to perform release
RESET_SERVO_AFTER_DROP = True  # whether to return servo to max() after drop

# ---- DroneController (same as earlier) ----
class DroneController:
    def __init__(self, system_address=SYSTEM_ADDRESS):
        if not MAVSDK_AVAILABLE:
            raise RuntimeError("mavsdk not available")
        self.system_address = system_address
        self.loop = None
        self.thread = None
        self._stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.drone = System()
        self._connected = False

    def start(self):
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        ok = self.ready_event.wait(timeout=30)
        if not ok:
            print("[WARN] DroneController not ready within 30s. Check connection.")
        else:
            print("[INFO] DroneController ready")

    def _thread_main(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._connect_and_setup())
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop=self.loop)
            for t in pending:
                t.cancel()
            try:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self.loop.close()

    async def _connect_and_setup(self):
        try:
            print(f"[DRONE] connecting to: {self.system_address} ...")
            await self.drone.connect(system_address=self.system_address)

            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    print("[DRONE] connected")
                    break

            print("[DRONE] arming...")
            await self.drone.action.arm()

            print("[DRONE] takeoff...")
            await self.drone.action.takeoff()
            time.sleep(10)
            await asyncio.sleep(4)

            await self.drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, TAKEOFF_ALT, 0.0))
            await self.drone.offboard.start()
            print("[DRONE] offboard started, holding position.")

            self._connected = True
            self.ready_event.set()
        except Exception as e:
            print("[DRONE] setup error:", e)
            self.ready_event.set()

    def forward(self, speed=0.5, duration=0.5):
        if not self.loop or not self._connected:
            print("[DRONE] forward requested but drone not ready")
            return
        fut = asyncio.run_coroutine_threadsafe(self._do_forward(speed, duration), self.loop)
        def _cb(f):
            try:
                f.result()
            except Exception as e:
                print("[DRONE] forward task error:", e)
        fut.add_done_callback(_cb)

    async def _do_forward(self, speed, duration):
        try:
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(speed, 0.0, 0.0, 0.0))
        except Exception as e:
            print("[DRONE] set_velocity_body error (start):", e)
            return
        await asyncio.sleep(duration)
        try:
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        except Exception as e:
            print("[DRONE] set_velocity_body error (stop):", e)

    def stop_and_land(self):
        if not self.loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._stop_and_land_coro(), self.loop)
        try:
            fut.result(timeout=20)
        except Exception as e:
            print("[DRONE] stop_and_land error:", e)
        def _stop_loop():
            self.loop.stop()
        self.loop.call_soon_threadsafe(_stop_loop)
        if self.thread:
            self.thread.join(timeout=5)

    async def _stop_and_land_coro(self):
        try:
            print("[DRONE] stopping offboard...")
            await self.drone.offboard.stop()
        except Exception as e:
            print("[DRONE] offboard stop error:", e)
        try:
            print("[DRONE] landing...")
            await self.drone.action.land()
        except Exception as e:
            print("[DRONE] landing error:", e)


# ---- Init cameras ----
cam0 = Picamera2(0)  # bottom
cam1 = Picamera2(1)  # front

cfg0 = cam0.create_preview_configuration(main={"format":"BGR888", "size": CAM_SIZE})
cfg1 = cam1.create_preview_configuration(main={"format":"BGR888", "size": CAM_SIZE})
cam0.configure(cfg0)
cam1.configure(cfg1)
cam0.start()
cam1.start()

# ---- Load model ----
model = YOLO(MODEL_PATH)

# ---- Video writer (optional) ----
writer = None
output_path = None

# ---- Servo setup ----
servo = None
if SERVO_AVAILABLE:
    try:
        servo = Servo(SERVO_PIN)
        print(f"[SERVO] Servo initialized on pin {SERVO_PIN}")
        if SERVO_INIT_MAX:
            try:
                servo.max()
                print("[SERVO] set to max() (initial)")
            except Exception as e:
                print("[SERVO] failed to set max():", e)
    except Exception as e:
        print("[SERVO] failed to initialize servo:", e)
        servo = None
        SERVO_AVAILABLE = False
else:
    print("[SERVO] gpiozero Servo not available — drop() will be a no-op or simulated print")


# ---- Drone controller + forward/drop action wiring ----
drone = None
if MAVSDK_AVAILABLE:
    try:
        drone = DroneController(system_address=SYSTEM_ADDRESS)
        drone.start()
    except Exception as e:
        print("[WARN] Failed to start DroneController:", e)
        drone = None
else:
    print("[WARN] MAVSDK not available — forward() will be a no-op.")


def forward():
    """Non-blocking forward call used by detection loop."""
    if drone:
        drone.forward(speed=0.5, duration=0.5)
    else:
        print("[ACTION] forward() (noop - drone not available)")


def drop():
    """
    Use servo.min() to trigger the drop.
    By default this will:
      - call servo.min()
      - hold for DROP_HOLD_SECONDS
      - optionally call servo.max() again to reset
    Adjust RESET_SERVO_AFTER_DROP if your release mechanism must remain in min() position.
    """
    if not SERVO_AVAILABLE or servo is None:
        print("[ACTION] drop() called but servo not available — implement actuator here")
        return

    try:
        print("[ACTION] drop(): moving servo to min()")
        servo.min()
    except Exception as e:
        print("[SERVO] error calling min():", e)
        return

    # Hold long enough for the mechanical release to occur
    time.sleep(DROP_HOLD_SECONDS)

    if RESET_SERVO_AFTER_DROP:
        try:
            print("[ACTION] drop(): resetting servo to max()")
            servo.max()
        except Exception as e:
            print("[SERVO] error calling max() after drop:", e)


# ---- helpers for detection drawing & area calc ----
def draw_results(img, result):
    try:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clsids = result.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        boxes = np.array(result.boxes.xyxy).astype(np.float32)
        confs = np.array(result.boxes.conf).astype(np.float32)
        clsids = np.array(result.boxes.cls).astype(np.int32)

    for (x1, y1, x2, y2), c, cid in zip(boxes, confs, clsids):
        try:
            label_name = model.names[int(cid)]
        except Exception:
            label_name = str(int(cid))
        label = f"{label_name} {c:.2f}"
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


def get_largest_person_area_ratio(result, orig_w, orig_h, conf_thresh=PERSON_CONF_THR):
    max_ratio = 0.0
    try:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clsids = result.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        boxes = np.array(result.boxes.xyxy).astype(np.float32)
        confs = np.array(result.boxes.conf).astype(np.float32)
        clsids = np.array(result.boxes.cls).astype(np.int32)

    if boxes.size == 0:
        return 0.0

    sx = orig_w / float(INFER_SIZE)
    sy = orig_h / float(INFER_SIZE)

    for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, clsids):
        is_person = False
        try:
            name = model.names[int(cid)].lower()
            if name == "person":
                is_person = True
        except Exception:
            if int(cid) == 0:
                is_person = True

        if not is_person or conf < conf_thresh:
            continue

        x1o = max(0.0, x1 * sx)
        y1o = max(0.0, y1 * sy)
        x2o = min(orig_w, x2 * sx)
        y2o = min(orig_h, y2 * sy)
        w = max(0.0, x2o - x1o)
        h = max(0.0, y2o - y1o)
        area = w * h
        ratio = area / (orig_w * orig_h) if orig_w * orig_h > 0 else 0.0
        if ratio > max_ratio:
            max_ratio = ratio

    return max_ratio


# ---- Main detection + action loop ----
fps = 0.0
last = time.time()
dropped = False

try:
    while True:
        frame0_rgb = cam0.capture_array()
        frame1_rgb = cam1.capture_array()

        frame0 = cv2.cvtColor(frame0_rgb, cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2BGR)

        vis0 = frame0.copy()
        vis1 = frame1.copy()

        f0_in = cv2.resize(frame0, (INFER_SIZE, INFER_SIZE))
        f1_in = cv2.resize(frame1, (INFER_SIZE, INFER_SIZE))

        results = model([f0_in, f1_in], imgsz=INFER_SIZE, verbose=False)

        if len(results) >= 1:
            draw_results(vis0, results[0])
        if len(results) >= 2:
            draw_results(vis1, results[1])

        largest_person_ratio = 0.0
        if len(results) >= 2:
            h1, w1 = frame1.shape[:2]
            largest_person_ratio = get_largest_person_area_ratio(results[1], w1, h1)
        else:
            largest_person_ratio = 0.0

        if largest_person_ratio <= 0.0:
            dropped = False
            forward()
        else:
            print(f"[DETECT] largest_person_ratio={largest_person_ratio:.3f}")
            if largest_person_ratio < PERSON_AREA_THRESHOLD:
                forward()
            else:
                if not dropped:
                    drop()
                    dropped = True

        # visualization
        target_h = 360
        h0, w0 = vis0.shape[:2]
        h1v, w1v = vis1.shape[:2]
        vis0 = cv2.resize(vis0, (int(w0 * target_h / h0), target_h))
        vis1 = cv2.resize(vis1, (int(w1v * target_h / h1v), target_h))
        combined = np.hstack((vis0, vis1))

        now = time.time()
        loop_time = now - last
        last = now
        cur_fps = 1.0 / loop_time if loop_time > 0 else 0.0
        fps = FPS_AVG_ALPHA * fps + (1-FPS_AVG_ALPHA) * cur_fps
        cv2.putText(combined, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow(DISPLAY_WINDOW, combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    if drone:
        print("[MAIN] stopping drone (offboard stop & land)...")
        try:
            drone.stop_and_land()
        except Exception as e:
            print("[MAIN] drone stop_and_land error:", e)

    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
