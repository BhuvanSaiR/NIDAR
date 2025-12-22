#!/usr/bin/env python3
"""
Record from two cameras automatically detected until Ctrl+C.

Usage:
  python3 record_auto_two_cams.py
Options:
  --out-dir DIR       Directory to save recordings (default ./recordings)
  --codec CODEC       FourCC codec (default MJPG)
  --fps FPS           Target FPS (default 30)
  --width W           Target width (default 1280)
  --height H          Target height (default 720)
  --probe-time SEC    Seconds to probe each device (default 1.0)
  --max-index N       Max numeric index to probe /dev/video0..N (default 8)
  --num-cams N        Number of cameras to record (default 2)
"""

import cv2
import time
import threading
import os
import signal
import argparse
from glob import glob
from datetime import datetime

stop_event = threading.Event()

def signal_handler(sig, frame):
    print("\nSIGINT received — stopping recording...")
    stop_event.set()

signal.signal(signal.SIGINT, signal_handler)

def probe_device(device, probe_time=1.0):
    """
    Try to open device (int index or '/dev/videoX') and read a frame within probe_time seconds.
    Return True if frame read succeeded.
    """
    try:
        # convert numeric strings to int where appropriate, but allow '/dev/videoX' as string
        dev = int(device) if isinstance(device, (str,)) and device.isdigit() else device
    except Exception:
        dev = device

    # Try V4L2 backend first on Linux (improves chance on Pi)
    try:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(dev)
    except Exception:
        try:
            cap = cv2.VideoCapture(dev)
        except Exception:
            return False

    if not cap.isOpened():
        return False

    deadline = time.time() + probe_time
    success = False
    # Try a few grabs/reads until deadline
    while time.time() < deadline:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            success = True
            break
        # short sleep before trying again
        time.sleep(0.05)

    cap.release()
    return success

def detect_cameras(num_cams=2, max_index=8, probe_time=1.0):
    """
    Detect camera device IDs. Returns list of device identifiers (int or '/dev/videoX' strings).
    Strategy:
      - gather /dev/video* entries
      - also test numeric indices 0..max_index
      - probe each candidate and return first `num_cams` that respond
    """
    candidates = []

    # add /dev/video* entries first (sorted)
    dev_paths = sorted(glob('/dev/video*'))
    for p in dev_paths:
        candidates.append(p)

    # then numeric indices (avoid duplicates of those already in dev_paths)
    # if /dev/videoN exists that'll also show up; cv2 accepts integer indices too.
    for i in range(0, max_index + 1):
        dev_path = f"/dev/video{i}"
        if dev_path in dev_paths:
            # prefer /dev/videoN path (already added)
            continue
        candidates.append(str(i))

    print(f"[detect] probing candidates: {candidates}")

    found = []
    for cand in candidates:
        if probe_device(cand, probe_time=probe_time):
            print(f"[detect] found working camera: {cand}")
            found.append(cand)
            if len(found) >= num_cams:
                break
        else:
            print(f"[detect] no response from: {cand}")

    return found

def open_capture(device, width=None, height=None, fps=None):
    try:
        dev = int(device) if isinstance(device, (str,)) and device.isdigit() else device
    except Exception:
        dev = device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {device}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return cap, actual_w, actual_h, actual_fps

def cam_worker(name, device, out_path, codec, width, height, fps):
    try:
        cap, w, h, actual_fps = open_capture(device, width, height, fps)
    except Exception as e:
        print(f"[{name}] ERROR opening camera {device}: {e}")
        return

    if actual_fps == 0.0 and fps:
        actual_fps = fps

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, actual_fps or 20.0, (w, h))
    if not writer.isOpened():
        print(f"[{name}] WARNING: VideoWriter failed with codec {codec}. Trying MJPG...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_path, fourcc, actual_fps or 20.0, (w, h))
        if not writer.isOpened():
            print(f"[{name}] ERROR: Could not open VideoWriter for {out_path}")
            cap.release()
            return

    print(f"[{name}] Recording {device} -> {out_path} ({w}x{h} @ {actual_fps:.1f} fps)")
    last_report = time.time()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue
        writer.write(frame)
        if time.time() - last_report >= 5.0:
            print(f"[{name}] still recording... ({datetime.now().isoformat(timespec='seconds')})")
            last_report = time.time()
    cap.release()
    writer.release()
    print(f"[{name}] Stopped and closed {out_path}")

def timestamp_str():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def main():
    parser = argparse.ArgumentParser(description="Auto-detect and record N cameras until Ctrl+C")
    parser.add_argument('--out-dir', default='./recordings')
    parser.add_argument('--codec', default='MJPG')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--probe-time', type=float, default=1.0, help='seconds to probe each device')
    parser.add_argument('--max-index', type=int, default=8, help='max /dev/video index to probe')
    parser.add_argument('--num-cams', type=int, default=2, help='how many cameras to auto-detect')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[main] detecting cameras...")
    cams = detect_cameras(num_cams=args.num_cams, max_index=args.max_index, probe_time=args.probe_time)
    if len(cams) < args.num_cams:
        print(f"[main] Found only {len(cams)} working camera(s). Need {args.num_cams}. Exiting.")
        return

    threads = []
    tstamp = timestamp_str()
    for idx, dev in enumerate(cams[:args.num_cams], start=1):
        outp = os.path.join(args.out_dir, f"cam{idx}_{tstamp}.avi")
        th = threading.Thread(target=cam_worker, args=(f"CAM{idx}", dev, outp, args.codec, args.width, args.height, args.fps), daemon=True)
        threads.append(th)
        th.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()

    print("[main] all done. Exiting.")

if __name__ == "__main__":
    main()
