#!/usr/bin/env python3
"""
mavsdk_survey_yolo.py

- Parse QGroundControl .plan, extract first geoFence polygon
- Create lawnmower transects inside polygon (spacing=35m) at altitude=50m
- Upload mission via MAVSDK (serial connection string from CLI)
- Arm, takeoff to 50m, start mission
- Run YOLO11-n (person-only) on Pi camera (capture ~50ms/frame)
- On person detection: pause mission, enter Offboard and issue body-frame velocities
  to yaw-to-center and move forward until person disappears, then resume mission.
"""
import argparse
import asyncio
import json
import math
import threading
import time
import subprocess
from queue import Queue, Empty

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate as shapely_rotate
from pyproj import Transformer

from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# ----------------- Configuration (tweak if needed) -----------------
ALTITUDE_M = 50.0
TRANSECT_SPACING_M = 35.0
CAM_DEVICE = 0              # /dev/video0
CAM_W = 1280
CAM_H = 720
CAM_FPS = 20                # ~50 ms capture interval
CAM_EXPOSURE_MS = 0.5       # target exposure (best-effort)
YOLO_CONF = 0.4
YOLO_CLASSES = [0]          # person-only (COCO class 0)
FORWARD_VEL_MPS = 1.0       # forward velocity while approaching
MAX_YAW_RATE_DEG_S = 30.0   # max yaw rate command while centering
CENTER_TOLERANCE_PX = CAM_W * 0.05  # within 5% of width -> considered centered
DEBOUNCE_LOST_SEC = 1.0     # consider person lost if not seen for this long
# -----------------------------------------------------------------

# Inter-thread communication
detection_queue = Queue()
stop_event = threading.Event()
last_detection = None
last_detection_time = 0.0
mission_progress_idx = 0  # updated by mission progress subscriber

# ----------------- Utilities: parse .plan and create transects -----------------
def parse_plan_polygon(plan_path):
    with open(plan_path, "r") as f:
        plan = json.load(f)
    gf = plan.get("geoFence", {})  # QGC uses geoFence.polygons
    polygons = gf.get("polygons", [])
    if not polygons:
        raise ValueError("No geoFence.polygons found in plan")
    poly0 = polygons[0]
    poly_points = poly0.get("polygon")
    if not poly_points:
        raise ValueError("No polygon points in geoFence.polygons[0]")
    pts_lonlat = []
    for p in poly_points:
        if len(p) < 2:
            continue
        # QGC historically uses [lat, lon] entries; detect order by magnitude
        a, b = p[0], p[1]
        if -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0:
            lat, lon = a, b
        elif -90.0 <= b <= 90.0 and -180.0 <= a <= 180.0:
            lat, lon = b, a
        else:
            raise ValueError(f"Unrecognized coord: {p}")
        pts_lonlat.append((lon, lat))
    return pts_lonlat  # list of (lon, lat)

def utm_transformer(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return Transformer.from_crs("epsg:4326", f"epsg:{epsg}", always_xy=True), f"epsg:{epsg}"

def build_transect_waypoints(lonlat_pts, spacing_m=TRANSECT_SPACING_M, alt_m=ALTITUDE_M):
    avg_lon = sum(l for l, _ in lonlat_pts) / len(lonlat_pts)
    avg_lat = sum(lat for _, lat in lonlat_pts) / len(lonlat_pts)
    transformer, epsg = utm_transformer(avg_lon, avg_lat)
    inv = Transformer.from_crs(epsg, "epsg:4326", always_xy=True)

    poly_xy = [transformer.transform(lon, lat) for lon, lat in lonlat_pts]
    poly = Polygon(poly_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)

    minx, miny, maxx, maxy = poly.bounds
    xs = []
    x = minx - spacing_m
    while x <= maxx + spacing_m:
        xs.append(x)
        x += spacing_m

    transects = []
    for x in xs:
        seg = LineString([(x, miny - 1000), (x, maxy + 1000)])
        inter = poly.intersection(seg)
        if inter.is_empty:
            continue
        if isinstance(inter, LineString):
            transects.append(inter)
        else:
            for part in inter:
                if isinstance(part, LineString):
                    transects.append(part)

    # sort by centroid and alternate direction
    transects = sorted(transects, key=lambda s: s.centroid.y)
    waypoints = []
    reverse = False
    for seg in transects:
        coords = list(seg.coords)
        if not coords:
            continue
        a = coords[0]; b = coords[-1]
        if reverse:
            a, b = b, a
        reverse = not reverse
        lon1, lat1 = inv.transform(a[0], a[1])
        lon2, lat2 = inv.transform(b[0], b[1])
        waypoints.append((lat1, lon1, alt_m))
        waypoints.append((lat2, lon2, alt_m))
    return waypoints

# ----------------- Camera/YOLO thread (runs in separate thread) -----------------
def try_set_camera_exposure(device_index, exposure_ms):
    ok = False
    try:
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{device_index}", "-c", "exposure_auto=1"], check=False)
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{device_index}", "-c", "exposure_absolute=1"], check=False)
        ok = True
    except Exception:
        pass
    try:
        cap = cv2.VideoCapture(device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_ms))
        cap.release()
        ok = True
    except Exception:
        pass
    return ok

def camera_yolo_thread(loop):
    global last_detection, last_detection_time
    print("[camera] loading YOLO model (yolo11n)...")
    model = YOLO("yolo11n.pt")  # ultralytics will download weights if needed
    cap = cv2.VideoCapture(CAM_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    try:
        ok = try_set_camera_exposure(CAM_DEVICE, CAM_EXPOSURE_MS)
        print("[camera] exposure set attempt:", ok)
    except Exception:
        pass

    interval = 1.0 / CAM_FPS
    while not stop_event.is_set():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        # run YOLO inference
        results = model(frame, imgsz=(CAM_W, CAM_H), conf=YOLO_CONF, classes=YOLO_CLASSES, verbose=False)
        found = False
        best = None
        # ultralytics results: results is iterable; each r has .boxes
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                # xyxy might be tensor; safely convert
                xyxy = np.array(b.xyxy[0].cpu()) if hasattr(b.xyxy[0], "cpu") else np.array(b.xyxy[0])
                conf = float(b.conf[0]) if hasattr(b.conf[0], "__float__") else float(b.conf)
                x1, y1, x2, y2 = map(float, xyxy)
                w = x2 - x1; h = y2 - y1
                cx = x1 + w / 2.0; cy = y1 + h / 2.0
                if (not found) or (conf > best[4]):
                    best = (cx, cy, w, h, conf)
                    found = True
        if found:
            last_detection = best
            last_detection_time = time.time()
            # push to queue in a non-blocking way
            try:
                detection_queue.put_nowait(best)
            except Exception:
                pass
        time_spent = time.time() - t0
        to_sleep = interval - time_spent
        if to_sleep > 0:
            time.sleep(to_sleep)
    cap.release()
    print("[camera] thread exiting")

# ----------------- MAVSDK / drone control -----------------
async def upload_mission(drone, waypoints):
    items = []
    for lat, lon, alt in waypoints:
        # MissionItem signature (positional) commonly works; we keep camera_action NONE
        items.append(MissionItem(
            lat,
            lon,
            alt,
            5.0,            # speed m/s (nominal between wp)
            True,           # is_fly_through
            float('nan'),   # gimbal_pitch_deg
            float('nan'),   # gimbal_yaw_deg
            MissionItem.CameraAction.NONE,
            float('nan'),   # loiter_time_s
            float('nan')    # camera_photo_interval_s
        ))
    plan = MissionPlan(items)
    print(f"[mav] uploading {len(items)} mission items...")
    res = await drone.mission.upload_mission(plan)
    print("[mav] upload result:", res)

async def arm_and_takeoff(drone, altitude_m):
    print("[mav] arming...")
    await drone.action.arm()
    print("[mav] taking off...")
    await drone.action.takeoff()
    # wait until reached altitude (relative)
    async for pos in drone.telemetry.position():
        rel_alt = pos.relative_altitude_m
        print(f"[mav] relative altitude: {rel_alt:.1f} m")
        if rel_alt >= altitude_m * 0.9:
            print("[mav] reached target altitude")
            break
        await asyncio.sleep(0.5)

# mission progress subscriber: updates global mission_progress_idx
async def mission_progress_subscriber(drone):
    global mission_progress_idx
    async for prog in drone.mission.mission_progress():
        mission_progress_idx = prog.current
        # small sleep to yield
        await asyncio.sleep(0.01)

async def approach_person_and_resume(drone):
    """
    This coroutine monitors detection_queue and when a detection occurs,
    it pauses the mission, enters offboard, issues velocity+yaw commands
    to keep person centered and move forward, then resumes mission at saved index.
    """
    global last_detection_time, last_detection
    while not stop_event.is_set():
        try:
            det = detection_queue.get(timeout=0.2)  # blocking for small time
            if det:
                # Save current mission index
                saved_idx = mission_progress_idx
                print(f"[monitor] detection -> pausing mission (saved idx={saved_idx})")
                await drone.mission.pause_mission()
                print("[monitor] mission paused (vehicle HOLD)")

                # Prepare Offboard: set a zero setpoint first
                try:
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await drone.offboard.start()
                    print("[monitor] offboard started")
                except OffboardError as e:
                    print("[monitor] failed to start offboard:", e)
                    # fallback: resume mission (can't approach)
                    try:
                        await drone.mission.set_current_mission_item(saved_idx)
                        await drone.mission.start_mission()
                    except Exception as ex:
                        print("[monitor] resume failed:", ex)
                    continue

                # approach loop: remain until DETECTION disappears for DEBOUNCE_LOST_SEC
                while (time.time() - last_detection_time) < DEBOUNCE_LOST_SEC and not stop_event.is_set():
                    det_local = last_detection
                    if det_local is None:
                        await asyncio.sleep(0.05)
                        continue
                    cx, cy, w, h, conf = det_local
                    dx = cx - (CAM_W / 2.0)
                    # convert pixel offset to yaw-rate command proportionally
                    # map dx/CAM_W -> yaw_rate (-MAX..+MAX)
                    yaw_rate = (dx / (CAM_W / 2.0)) * MAX_YAW_RATE_DEG_S
                    yaw_rate = max(-MAX_YAW_RATE_DEG_S, min(MAX_YAW_RATE_DEG_S, yaw_rate))
                    # forward speed - simple constant; reduce if bbox large
                    fwd = FORWARD_VEL_MPS
                    # If target close (large bbox width), slow down
                    if w >= CAM_W * 0.4:
                        fwd = 0.2
                    try:
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(fwd, 0.0, 0.0, float(yaw_rate)))
                    except OffboardError as e:
                        print("[monitor] offboard set velocity error:", e)
                        break
                    await asyncio.sleep(0.2)
                # stop motion
                try:
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await drone.offboard.stop()
                    print("[monitor] offboard stopped")
                except OffboardError as e:
                    print("[monitor] error stopping offboard:", e)

                # resume mission where left off
                try:
                    print(f"[monitor] setting mission item to {saved_idx} and resuming mission")
                    await drone.mission.set_current_mission_item(saved_idx)
                    await drone.mission.start_mission()
                    print("[monitor] mission resumed")
                except Exception as e:
                    print("[monitor] failed to resume mission:", e)
        except Empty:
            await asyncio.sleep(0.05)

# ----------------- Main entrypoint -----------------
async def main_async(plan_path, connect_str):
    global last_detection, last_detection_time

    print("[main] parsing plan:", plan_path)
    pts = parse_plan_polygon(plan_path)
    print(f"[main] polygon points loaded: {len(pts)}")

    print("[main] building transect waypoints...")
    wps = build_transect_waypoints(pts, spacing_m=TRANSECT_SPACING_M, alt_m=ALTITUDE_M)
    if not wps:
        raise SystemExit("No waypoints generated")
    print(f"[main] generated {len(wps)} waypoint entries")

    print(f"[mav] connecting to {connect_str} ...")
    drone = System()
    await drone.connect(system_address=connect_str)

    # wait for connection
    print("[mav] waiting for system to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"[mav] connected to system (UUID: {state.uuid})")
            break
        await asyncio.sleep(0.1)

    # start mission progress subscriber
    asyncio.create_task(mission_progress_subscriber(drone))

    # upload mission
    try:
        await upload_mission(drone, wps)
    except Exception as e:
        print("[mav] mission upload failed:", e)
        return

    # arm & takeoff
    try:
        await arm_and_takeoff(drone, ALTITUDE_M)
    except Exception as e:
        print("[mav] takeoff failed:", e)
        return

    # start the mission
    try:
        print("[mav] starting mission")
        await drone.mission.start_mission()
    except Exception as e:
        print("[mav] start mission failed:", e)
        return

    # start camera/YOLO thread
    loop = asyncio.get_running_loop()
    cam_thread = threading.Thread(target=camera_yolo_thread, args=(loop,), daemon=True)
    cam_thread.start()
    print("[main] camera thread started")

    # start monitor task to react to detections
    monitor_task = asyncio.create_task(approach_person_and_resume(drone))

    # keep running until mission completes or keyboard interrupt
    try:
        # monitor in-air and mission progress to know when mission done
        mission_total = None
        async for prog in drone.mission.mission_progress():
            mission_total = prog.total
            print(f"[status] mission progress: {prog.current}/{prog.total} | mode: (n/a in MAVSDK) ")
            # if mission done (current == total)
            if prog.total != 0 and prog.current >= prog.total:
                print("[main] mission finished")
                break
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        print("[main] stopping camera thread and monitor")
        await asyncio.sleep(0.5)
        # attempt to land safely
        try:
            print("[mav] landing...")
            await drone.action.land()
        except Exception as e:
            print("[mav] landing failed:", e)
        await asyncio.sleep(2.0)

def main():
    parser = argparse.ArgumentParser(description="MAVSDK survey with YOLO person interrupt")
    parser.add_argument("plan", help=".plan file path")
    parser.add_argument("--connect", required=True, help="MAVSDK connection string e.g. serial:///dev/ttyACM0:115200")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args.plan, args.connect))
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
