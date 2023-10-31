import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/derrick/data/speedplusv2/synthetic/images/img000001.jpg')

# Convert the image to grayscale & blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# Find and fill contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Save / process image
cv2.imwrite('outer_boundary_mask.png', mask)
cv2.imshow('Outer Boundary Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()





