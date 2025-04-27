import cv2
import numpy as np
import pytesseract
from PIL import Image
import json
import os

# Path to the folder where day images are saved
days_folder = 'days'

# Clear out the 'days' folder before starting
if os.path.exists(days_folder):
    for filename in os.listdir(days_folder):
        file_path = os.path.join(days_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(days_folder)

# Load the image
img = cv2.imread('calendar.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to get binary image
thresh = cv2.adaptiveThreshold(gray, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 15, 10)

# Detect horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detect_horizontal = cv2.erode(thresh, horizontal_kernel, iterations=2)
horizontal_lines = cv2.dilate(detect_horizontal, horizontal_kernel, iterations=2)

# Detect vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
detect_vertical = cv2.erode(thresh, vertical_kernel, iterations=2)
vertical_lines = cv2.dilate(detect_vertical, vertical_kernel, iterations=2)

# Combine lines
grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

# Find intersections
intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)

# Find intersection points
contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get centers of intersection points
centers = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    centers.append((x + w//2, y + h//2))

# Sort centers
centers = sorted(centers, key=lambda c: (c[1], c[0]))  # sort by y, then x

# Group centers by rows
tolerance = 20  # pixels
rows = []
current_row = []
for center in centers:
    if not current_row:
        current_row.append(center)
    else:
        if abs(center[1] - current_row[-1][1]) < tolerance:
            current_row.append(center)
        else:
            rows.append(current_row)
            current_row = [center]
if current_row:
    rows.append(current_row)

# Sort each row by x
for row in rows:
    row.sort(key=lambda c: c[0])

# Now slice each cell
calendar_data = {"days": []}
day_num = 1

# You have (n-1) cells per (n) points
for i in range(len(rows) - 1):
    min_length = min(len(rows[i]), len(rows[i+1])) - 1
    for j in range(min_length):
        x1, y1 = rows[i][j]
        x2, y2 = rows[i+1][j+1]

        # Crop the cell
        cell = img[y1:y2, x1:x2]

        # Save image
        filename = f"days/day_{day_num:02d}.jpg"
        cv2.imwrite(filename, cell)

        # OCR the cell
        text = pytesseract.image_to_string(cell, config='--psm 6')
        text = text.strip()

        calendar_data["days"].append({
            "day": day_num,
            "events": text
        })

        day_num += 1

# Save to JSON
with open('calendar_events.json', 'w', encoding='utf-8') as f:
    json.dump(calendar_data, f, indent=2)

print("Done! Saved all day images and extracted events.")