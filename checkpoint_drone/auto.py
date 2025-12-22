#!/usr/bin/env python3
"""
human_center_and_approach_picamera2_takeoff_preview.py (modified for ArduPilot)

- Uses a robust arm/takeoff + offboard priming routine that sends multiple initial zero setpoints
  before starting offboard (this is required/recommended for ArduPilot).
- Takes off to TAKEOFF_ALT_M (with timeout) and then runs the same Picamera2 + YOLO loop
  with live preview and person-follow behavior.
"""

import asyncio
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2

from picamera2 import Picamera2
from ultralytics import YOLO

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# ---------------- USER CONFIG ----------------
SYSTEM_ADDR = "serial:///dev/ttyACM0:115200"   # MAVSDK connection string
CAMERA_ID = 0                  # Picamera2 constructor param (0 or 1 on some boards)
EXPOSURE_TIME_MS = 0.5         # in milliseconds (0.5 ms -> 500 us)
ISO = 100                      # approximate; AnalogueGain = ISO / 100
INTERVAL_MS = 100              # capture interval in milliseconds
YOLO_MODEL = "yolo11n.pt"      # ultralytics model (nano)
CONF_THRESH = 0.35
CENTER_THRESH_PIX = 30         # px tolerance to consider centered
YAW_KP = 40.0                  # proportional gain (normalized offset -> deg/s)
MAX_YAW_RATE = 20.0            # deg/s (slow)
FORWARD_SPEED_M_S = 1.0
TAKEOFF_ALT_M = 2.0            # takeoff target (meters)
TAKEOFF_TIMEOUT_S = 15         # seconds to wait for altitude
OFFBOARD_PRIME_COUNT = 20      # how many zero setpoints to send before offboard.start()
OFFBOARD_PRIME_INTERVAL_S = 0.1
PREVIEW_WINDOW = "YOLO Preview"
# ----------------------------------------------

@dataclass
class FrameDetection:
    bbox: np.ndarray  # [x1,y1,x2,y2]
    conf: float
    class_id: int

def setup_picamera2(camera_id: int, exposure_time_ms: float, iso: int):
    picam2 = Picamera2(camera_id)
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    exposure_us = int(exposure_time_ms * 1000.0)  # ms -> us
    analogue_gain = float(iso) / 100.0

    controls = {
        "ExposureTime": exposure_us,
        "AnalogueGain": analogue_gain,
    }
    try:
        picam2.set_controls(controls)
        print(f"[camera] set controls ExposureTime={exposure_us}us AnalogueGain={analogue_gain}")
    except Exception as e:
        print("[camera] Warning: could not set controls exactly. Camera/driver may not support them:", e)

    # grab one frame to determine shape
    try:
        frame = picam2.capture_array()
        h, w = frame.shape[:2]
    except Exception:
        w, h = 640, 480

    return picam2, w, h

def draw_detections_bgr(bgr_img, detections):
    """Draw bounding boxes & labels on a BGR image (in-place)."""
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        conf = d.conf
        label = f"person {conf:.2f}"
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(bgr_img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(bgr_img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

async def safe_arm_takeoff_and_start_offboard(drone: System,
                                             takeoff_alt_m: float = 1.0,
                                             takeoff_timeout: float = 15.0,
                                             initial_setpoints: int = 20,
                                             setpoint_interval_s: float = 0.1):
    """
    Attempts to arm, takeoff, wait up to takeoff_timeout for altitude,
    then sends a burst of zero offboard setpoints and starts offboard.
    This is more robust for ArduPilot which likes multiple preceding setpoints.
    Returns True if offboard.start() succeeded, False otherwise.
    """
    # 1) Arm
    try:
        print("[mavsdk] Arming...")
        await drone.action.arm()
    except Exception as e:
        print("[mavsdk] Warning: arm() raised:", e)

    # 2) set takeoff altitude if available
    try:
        await drone.action.set_takeoff_altitude(takeoff_alt_m)
    except Exception:
        pass

    # 3) Takeoff command
    try:
        print(f"[mavsdk] Taking off to ~{takeoff_alt_m} m ...")
        await drone.action.takeoff()
    except Exception as e:
        print("[mavsdk] Warning: takeoff() raised:", e)

    # 4) Wait for altitude with timeout (robust to missing fields)
    print(f"[mavsdk] Waiting up to {takeoff_timeout}s for altitude >= {takeoff_alt_m*0.9} m ...")
    start_wait = time.time()
    reached = False
    try:
        async for pos in drone.telemetry.position():
            alt = getattr(pos, "relative_altitude_m", None)
            if alt is not None:
                print(f"[telemetry] relative altitude: {alt:.2f} m")
                if alt >= 0.9 * takeoff_alt_m:
                    reached = True
                    break
            # break on timeout
            if time.time() - start_wait > takeoff_timeout:
                print("[mavsdk] altitude wait timed out")
                break
    except Exception as e:
        print("[mavsdk] telemetry.position() iteration error (continuing):", e)

    if reached:
        print("[mavsdk] Takeoff altitude reached.")
    else:
        print("[mavsdk] Takeoff altitude NOT confirmed; proceeding anyway (timeout or telemetry missing).")

    # 5) Send multiple initial zero setpoints BEFORE enabling offboard
    print(f"[mavsdk] Sending {initial_setpoints} initial zero offboard setpoints (interval {setpoint_interval_s}s)...")
    zero_sp = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    for i in range(initial_setpoints):
        try:
            await drone.offboard.set_velocity_body(zero_sp)
            if (i + 1) % 5 == 0:
                print(f"[mavsdk] primed {i+1}/{initial_setpoints}")
        except Exception as e:
            # log but continue trying — some implementations accept setpoints even if offboard not yet started
            print(f"[mavsdk] Warning: set_velocity_body during priming failed (i={i}): {e}")
        await asyncio.sleep(setpoint_interval_s)

    # 6) Try to start offboard
    try:
        print("[mavsdk] Starting offboard...")
        await drone.offboard.start()
        print("[mavsdk] Offboard started.")
        return True
    except Exception as e:
        print("[mavsdk] Offboard start failed (exception):", e)
        return False

async def main():
    # Load model
    print("[model] Loading YOLO model...")
    model = YOLO(YOLO_MODEL)  # may download if not present; CPU by default

    # Connect to drone
    drone = System()
    print(f"[mavsdk] Connecting to {SYSTEM_ADDR} ...")
    await drone.connect(system_address=SYSTEM_ADDR)

    print("[mavsdk] Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- connected to drone")
            break

    # Use the robust arm/takeoff/offboard priming routine
    offboard_started = await safe_arm_takeoff_and_start_offboard(drone,
                                                                 takeoff_alt_m=TAKEOFF_ALT_M,
                                                                 takeoff_timeout=TAKEOFF_TIMEOUT_S,
                                                                 initial_setpoints=OFFBOARD_PRIME_COUNT,
                                                                 setpoint_interval_s=OFFBOARD_PRIME_INTERVAL_S)
    if not offboard_started:
        print("[mavsdk] Warning: Offboard did not start cleanly. The script will continue but offboard commands may be ignored by the autopilot.")

    # Setup Picamera2 and compute frame center
    picam2, width, height = setup_picamera2(CAMERA_ID, EXPOSURE_TIME_MS, ISO)
    frame_center_x = width / 2.0
    frame_center_y = height / 2.0
    print(f"[camera] frame size ~ {width}x{height}")

    # Ensure initial offboard zero setpoint (attempt one more time)
    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    except Exception as e:
        print("[mavsdk] Warning: initial offboard setpoint failed:", e)

    print("[main] Entering capture+detect loop. Press 'q' in the preview window to quit.")

    frame_count = 0
    try:
        while True:
            # timing to target interval
            target_time = frame_count * (INTERVAL_MS / 1000.0)
            # use loop start time relative
            if frame_count == 0:
                loop_start = time.perf_counter()
            now = time.perf_counter()
            sleep_for = (loop_start + target_time) - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

            # capture frame (Picamera2 returns RGB)
            try:
                rgb_frame = await asyncio.to_thread(picam2.capture_array)
            except Exception as e:
                print("[camera] capture_array failed:", e)
                await asyncio.sleep(0.01)
                frame_count += 1
                continue

            frame_count += 1

            # Prepare a BGR copy for preview drawing
            try:
                bgr_for_display = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            except Exception:
                # if already BGR or conversion failed, fallback
                bgr_for_display = rgb_frame.copy()

            # run model inference (on RGB) in a thread
            try:
                results = await asyncio.to_thread(model.predict, source=rgb_frame, imgsz=640, conf=CONF_THRESH, device="cpu", stream=False)
            except Exception as e:
                # fallback API shape
                try:
                    results = await asyncio.to_thread(model.predict, rgb_frame, 640, CONF_THRESH, False)
                except Exception as e2:
                    print("[model] inference failed:", e2)
                    results = []

            # parse detections
            detections = []
            if len(results) > 0:
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    for i in range(len(r.boxes)):
                        try:
                            xyxy = r.boxes.xyxy[i].cpu().numpy()
                            conf = float(r.boxes.conf[i].cpu().numpy())
                            cls = int(r.boxes.cls[i].cpu().numpy())
                        except Exception:
                            # some ultralytics versions return numpy-like arrays already
                            xyxy = np.array(r.boxes.xyxy[i])
                            conf = float(r.boxes.conf[i])
                            cls = int(r.boxes.cls[i])
                        detections.append(FrameDetection(bbox=xyxy, conf=conf, class_id=cls))

            # filter persons (COCO person class 0)
            persons = [d for d in detections if d.class_id == 0 and d.conf >= CONF_THRESH]

            # draw detections on preview
            draw_detections_bgr(bgr_for_display, persons)

            # show center crosshair
            cv2.drawMarker(bgr_for_display, (int(frame_center_x), int(frame_center_y)), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.imshow(PREVIEW_WINDOW, bgr_for_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[main] Quit requested by user via 'q' key.")
                break

            # If no person, send zero velocities (hover)
            if len(persons) == 0:
                try:
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                except Exception as e:
                    print("[mavsdk] Warning: set_velocity_body failed (no person):", e)
                continue

            # choose largest bbox (heuristic for closest)
            def area(d):
                x1, y1, x2, y2 = d.bbox
                return max(0.0, (x2-x1)*(y2-y1))
            persons.sort(key=area, reverse=True)
            target = persons[0]
            cx, cy = bbox_center(target.bbox)
            dx = cx - frame_center_x    # positive = right
            norm_dx = dx / frame_center_x
            yaw_rate = float(np.clip(norm_dx * YAW_KP, -MAX_YAW_RATE, MAX_YAW_RATE))

            # if not centered, yaw only
            if abs(dx) > CENTER_THRESH_PIX:
                try:
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, yaw_rate))
                except Exception as e:
                    print("[mavsdk] Warning: set_velocity_body failed (yaw):", e)
                print(f"[action] yawing: yaw_rate={yaw_rate:.2f} deg/s  dx={dx:.1f}px")
                continue

            # centered -> approach forward until person leaves or drifts
            print(f"[action] centered (dx={dx:.1f}px). Approaching forward at {FORWARD_SPEED_M_S} m/s")
            while True:
                try:
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(FORWARD_SPEED_M_S, 0.0, 0.0, 0.0))
                except Exception as e:
                    print("[mavsdk] Warning: set_velocity_body failed (forward):", e)
                await asyncio.sleep(INTERVAL_MS / 1000.0)

                # capture a fresh frame
                try:
                    rgb2 = await asyncio.to_thread(picam2.capture_array)
                except Exception:
                    break

                # preview update (draw quickly)
                try:
                    bgr2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR)
                except Exception:
                    bgr2 = rgb2.copy()

                # inference
                try:
                    res2 = await asyncio.to_thread(model.predict, source=rgb2, imgsz=640, conf=CONF_THRESH, device="cpu", stream=False)
                except Exception:
                    try:
                        res2 = await asyncio.to_thread(model.predict, rgb2, 640, CONF_THRESH, False)
                    except Exception:
                        res2 = []

                persons2 = []
                if len(res2) > 0:
                    r2 = res2[0]
                    if hasattr(r2, "boxes") and r2.boxes is not None and len(r2.boxes) > 0:
                        for i in range(len(r2.boxes)):
                            try:
                                xy2 = r2.boxes.xyxy[i].cpu().numpy()
                                conf2 = float(r2.boxes.conf[i].cpu().numpy())
                                cls2 = int(r2.boxes.cls[i].cpu().numpy())
                            except Exception:
                                xy2 = np.array(r2.boxes.xyxy[i])
                                conf2 = float(r2.boxes.conf[i])
                                cls2 = int(r2.boxes.cls[i])
                            if cls2 == 0 and conf2 >= CONF_THRESH:
                                persons2.append(FrameDetection(bbox=xy2, conf=conf2, class_id=cls2))

                draw_detections_bgr(bgr2, persons2)
                cv2.drawMarker(bgr2, (int(frame_center_x), int(frame_center_y)), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
                cv2.imshow(PREVIEW_WINDOW, bgr2)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    print("[main] Quit requested by user during approach.")
                    raise KeyboardInterrupt()

                if len(persons2) == 0:
                    print("[action] person left frame. Stopping forward motion.")
                    try:
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    except Exception as e:
                        print("[mavsdk] Warning: set_velocity_body failed (stop):", e)
                    break

                persons2.sort(key=area, reverse=True)
                tgt2 = persons2[0]
                cx2, cy2 = bbox_center(tgt2.bbox)
                dx2 = cx2 - frame_center_x
                if abs(dx2) > CENTER_THRESH_PIX:
                    print("[action] drifted while approaching; will re-center")
                    # break to main loop to handle yaw re-centering
                    break

            # small pause then continue main loop
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("Exit requested (KeyboardInterrupt).")
    finally:
        print("[cleanup] stopping offboard & landing (if possible)...")
        try:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.2)
            await drone.offboard.stop()
        except Exception as e:
            print("[cleanup] Offboard stop/zero failed:", e)

        # command landing
        try:
            print("[cleanup] Landing...")
            await drone.action.land()
            await asyncio.sleep(5.0)
        except Exception as e:
            print("[cleanup] Landing command failed or not supported:", e)

        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Program terminated.")

if __name__ == "__main__":
    asyncio.run(main())
