# save_as: capture_picamera2_30ms_100ms_interval.py
# Requires: picamera2 (pip package on Raspberry Pi OS with libcamera)
# Notes:
# - ExposureTime is given in microseconds for libcamera controls (30 ms -> 30000).
# - AnalogueGain is a rough proxy for ISO. Many users use AnalogueGain ≈ ISO/100.
# - IO/write speed and camera internal processing may mean the real interval won't be exactly 100 ms.

from picamera2 import Picamera2
import time
from pathlib import Path

# --- user parameters ---
exposure_time_ms = 10        # desired exposure in milliseconds
iso = 800                    # desired ISO (approx — will set analogue gain = iso / 100)
interval_ms = 100            # capture interval in milliseconds
num_frames = 20
output_dir = Path("captures_10ms")
output_dir.mkdir(parents=True, exist_ok=True)

# --- derived controls ---
exposure_us = int(exposure_time_ms * 1000)            # libcamera expects microseconds
analogue_gain = float(iso) / 100.0                    # approximate mapping: ISO 100 -> gain 1.0

# --- camera setup ---
picam2 = Picamera2()
config = picam2.create_still_configuration()          # use still config (adjust if you want raw)
picam2.configure(config)
picam2.start()

# Apply manual controls. Some cameras/drivers may require switching off auto-exposure first.
controls = {
    "ExposureTime": exposure_us,      # microseconds
    "AnalogueGain": analogue_gain,    # approximate gain (float)
    # you can add more controls if needed, e.g. "AwbEnable": False, "ColourGains": (1.0,1.0)
}
try:
    picam2.set_controls(controls)
except Exception as e:
    print("Warning: could not set controls exactly. Camera/driver may not support them:", e)

print(f"Starting capture: {num_frames} frames, {exposure_time_ms} ms exposure, every {interval_ms} ms")

start_time = time.perf_counter()
for i in range(num_frames):
    frame_index = i + 1
    # filename
    fname = output_dir / f"frame_{frame_index:03d}.jpg"
    # capture to file (blocking)
    try:
        picam2.capture_file(str(fname))
    except Exception as e:
        print(f"Error capturing frame {frame_index}: {e}")
        break

    # compute next target time and sleep (accurate-ish scheduling)
    next_target = start_time + (frame_index * (interval_ms / 1000.0))
    now = time.perf_counter()
    sleep_for = next_target - now
    if sleep_for > 0:
        time.sleep(sleep_for)
    else:
        # we are behind schedule; continue immediately
        pass

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Finished {i+1} frames in {elapsed:.3f}s (avg interval {elapsed/(i+1):.3f}s)")

picam2.stop()
