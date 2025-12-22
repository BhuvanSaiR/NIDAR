from picamera2 import Picamera2
import cv2
import time

def main():
    picam2 = Picamera2()

    # Create a standard preview configuration
    config = picam2.create_preview_configuration()
    picam2.configure(config)

    picam2.start()
    time.sleep(0.5)  # small delay to let the camera warm up

    window_name = "Raspberry Pi Camera Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Press 'q' or ESC to quit.")

    try:
        while True:
            # Capture a frame as a NumPy array (RGB)
            frame = picam2.capture_array()

            # OpenCV expects BGR, so convert
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow(window_name, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
