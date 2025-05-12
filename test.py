from calendar_cropper import crop_calendar_region
from PIL import Image
import numpy as np

image = np.array(Image.open("calendar-may.jpg"))
box = crop_calendar_region(image)
cropped = Image.fromarray(image).crop(box)
cropped.save("cropped_example.jpg")
