import cv2
import time
from picamera2 import Picamera2
import numpy as np
from datetime import datetime

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

time.sleep(2)  # Let camera warm up

# Initialize background frame
prev_frame = None
motion_detected = False
motion_timer = None
motion_timeout = 5  # seconds of stillness before capturing

print("Monitoring for motion...")

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    # Compute difference between current frame and previous
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    motion_score = np.sum(thresh) / 255

    if motion_score > 100:  # tune this threshold
        if not motion_detected:
            print("Motion detected!")
        motion_detected = True
        motion_timer = time.time()  # reset timer
    else:
        if motion_detected and motion_timer and (time.time() - motion_timer) > motion_timeout:
            print("Motion stopped. Capturing image...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            picam2.capture_file(f"./motion_{timestamp}.jpg")
            print(f"Image saved as motion_{timestamp}.jpg")
            motion_detected = False
            motion_timer = None

    prev_frame = gray
    time.sleep(0.2)
