import cv2
import numpy as np
import pytesseract
from PIL import Image
import json
import os

# Make sure Tesseract is installed and properly referenced
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # if needed

# Load the calendar image
img = cv2.imread('calendar.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# Detect edges and lines
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Draw detected lines
line_img = np.zeros_like(img)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_img, (x1, y1), (x2, y2), (255,255,255), 2)

# Find cell contours
gray_lines = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prepare output
calendar_data = {"days": []}

# Create output folder for day images
output_folder = "days"
os.makedirs(output_folder, exist_ok=True)

# Sort contours top to bottom, left to right
bounding_boxes = [cv2.boundingRect(c) for c in contours]
bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))  # sort by y, then x

i = 1
for x, y, w, h in bounding_boxes:
    if w > 50 and h > 50:  # filter small noise
        day_img = img[y:y+h, x:x+w]
        
        # Save each cropped day image
        day_filename = os.path.join(output_folder, f"day_{i:02d}.jpg")
        cv2.imwrite(day_filename, day_img)

        # OCR on the saved day image
        text = pytesseract.image_to_string(Image.open(day_filename), config='--psm 6')
        text = text.strip()

        calendar_data["days"].append({
            "day": i,
            "events": text
        })
        i += 1

# Save to JSON
with open('calendar_events.json', 'w', encoding='utf-8') as f:
    json.dump(calendar_data, f, indent=2)

print("Calendar events saved to calendar_events.json!")
print(f"Cropped day images saved to {output_folder}/")