import cv2
import time
import numpy as np
from picamera2 import Picamera2
from PIL import Image
from datetime import datetime

# Initialize Picamera2
picam2 = Picamera2()

# Low-res video config for motion detection
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(video_config)
picam2.start()
time.sleep(2)  # Camera warm-up

# Autofocus once and lock (optional)
try:
    picam2.set_controls({"AfMode": 2})  # Autofocus once
    time.sleep(1)
    picam2.set_controls({"AfMode": 0})  # Lock focus
except Exception as e:
    print("Autofocus not supported or failed:", e)

# Motion detection state
prev_frame = None
motion_detected = False
motion_timer = None
motion_timeout = 2  # Seconds of stillness before taking photo
motion_threshold = 100  # Lower = more sensitive

print("Monitoring for motion...")

while True:
    # Capture frame and ensure it's usable
    frame = picam2.capture_array("main").copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    # Frame diff for motion detection
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    motion_score = np.sum(thresh) / 255

    if motion_score > motion_threshold:
        if not motion_detected:
            print("Motion detected!")
        motion_detected = True
        motion_timer = time.time()
    else:
        if motion_detected and motion_timer and (time.time() - motion_timer) > motion_timeout:
            print("Motion stopped. Preparing to capture image...")
            time.sleep(1.5)  # Let the scene settle

            # Switch to still mode
            picam2.stop()
            still_config = picam2.create_still_configuration(main={"size": (3280, 2464)})
            picam2.configure(still_config)
            picam2.start()
            time.sleep(1)  # Let auto settings stabilize

            # Autofocus again (if supported)
            try:
                picam2.set_controls({"AfMode": 2})
                time.sleep(2)
                picam2.set_controls({"AfMode": 0})
            except:
                pass

            # Optional: manual exposure/gain (sharper images)
            try:
                picam2.set_controls({
                    "ExposureTime": 0,     # 0 tells it to go auto
                    "AnalogueGain": 0.0    # 0.0 = auto gain
                })
            except:
                pass

            time.sleep(0.5)  # Final pause before capture

            # Capture image and save
            image = picam2.capture_array("main").copy()
            timestamp = datetime.now().strftime("%m-%d-%Y (%H-%M-%S)")
            filename = f"/mnt/mom-calendar/{timestamp}.jpg"

            # Save full image
            Image.fromarray(image).rotate(180).save(filename)
            print(f"Image saved: {filename}")

            # Automatically crop calendar area and save
            def crop_calendar_region(img):
                import cv2

                # Convert to grayscale and blur to reduce noise
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # Edge detection
                edged = cv2.Canny(blurred, 50, 150)

                # Find contours
                contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                best_box = None
                max_area = 0

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    aspect = w / h if h != 0 else 0

                    # Filter likely calendar shapes
                    if area > 10000 and 0.5 < aspect < 2.0:
                        if area > max_area:
                            best_box = (x, y, x + w, y + h)
                            max_area = area

                if best_box:
                    print("Calendar region found:", best_box)
                    return best_box
                else:
                    print("No calendar region found. Returning full image.")
                    return (0, 0, img.shape[1], img.shape[0])  # fallback: full image

            calendar_crop_box = crop_calendar_region(image)
            calendar_image = Image.fromarray(image).crop(calendar_crop_box)
            calendar_filename = filename.replace(".jpg", "_calendar.jpg")
            calendar_image.save(calendar_filename)
            print(f"Cropped calendar image saved: {calendar_filename}")

            # Return to video mode for motion detection
            picam2.stop()
            picam2.configure(video_config)
            picam2.start()
            time.sleep(1)

            motion_detected = False
            motion_timer = None

    prev_frame = gray
    time.sleep(0.2)
