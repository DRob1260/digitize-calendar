import cv2

def crop_calendar_region(img):
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
        return (0, 0, img.shape[1], img.shape[0])