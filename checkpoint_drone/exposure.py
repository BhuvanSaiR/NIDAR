from picamera2 import Picamera2
from time import sleep, time

picam2 = Picamera2()

# --- Camera configuration for still captures ---
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()

# Give sensor a short moment to stabilize
sleep(0.2)

# --- Set fixed exposure (10 microseconds) ---
picam2.set_controls({
    "AeEnable": False,
    "ExposureTime": 10,      # 10 s exposure
    "AnalogueGain": 1.0
})

print("Starting capture loop: 10 s exposure, saving every 100 ms...")

interval = 0.1       # 100 ms
count = 1

while True:
    start = time()

    filename = f"image_{count:04d}.jpg"
    picam2.capture_file(filename)
    print(f"Saved {filename}")

    count += 1

    # Sleep only the remaining time to maintain 100 ms period
    elapsed = time() - start
    if elapsed < interval:
        sleep(interval - elapsed)
