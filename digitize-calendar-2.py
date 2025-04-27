import cv2
import numpy as np
import pytesseract
from PIL import Image
import json
import os

# Load image
img = cv2.imread('calendar-3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest rectangular contour is the calendar
contours = sorted(contours, key=cv2.contourArea, reverse=True)
calendar_contour = contours[0]

# Approximate to a rectangle
peri = cv2.arcLength(calendar_contour, True)
approx = cv2.approxPolyDP(calendar_contour, 0.02 * peri, True)

if len(approx) == 4:
    # Perspective transform to get top-down view
    pts = approx.reshape(4, 2)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect

    # Compute width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Now split into 7 columns and 5 (or 6) rows
    rows = 6  # some months may need 5 or 6
    cols = 7

    cell_width = maxWidth // cols
    cell_height = maxHeight // rows

    # Create output folder
    output_folder = "days"
    os.makedirs(output_folder, exist_ok=True)

    calendar_data = {"days": []}
    day_num = 1

    for row in range(rows):
        for col in range(cols):
            x = col * cell_width
            y = row * cell_height
            day_crop = warp[y:y+cell_height, x:x+cell_width]

            # Save each day
            day_filename = os.path.join(output_folder, f"day_{day_num:02d}.jpg")
            cv2.imwrite(day_filename, day_crop)

            # OCR
            text = pytesseract.image_to_string(day_crop, config='--psm 6')
            text = text.strip()

            calendar_data["days"].append({
                "day": day_num,
                "events": text
            })

            day_num += 1

    # Save JSON
    with open('calendar_events.json', 'w', encoding='utf-8') as f:
        json.dump(calendar_data, f, indent=2)

    print("Done! Saved day images and events.")
else:
    print("Could not find a rectangular calendar!")