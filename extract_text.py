# Library imports
from PIL import Image
import pytesseract
import cv2
import os

# Load image
image = cv2.imread('./images/bar.jpg')

# Grey scale it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", gray)
cv2.waitKey(0)

# Run pytesseract
text = pytesseract.image_to_string(gray)
print(text)