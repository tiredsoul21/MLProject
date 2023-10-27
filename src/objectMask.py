import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/derrick/data/speedplusv2/synthetic/images/img000001.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use canny for edge detection
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# Apply morphological to help connect stuff
kernel = np.ones((30, 30), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create mask
mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Save / process the image
cv2.imwrite('object_mask.png', mask)
cv2.imshow('Connected Boundary Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
