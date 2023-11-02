import os

import cv2
import numpy as np


# Scale all images in a directory to the range [0, 1]
def maskImageDir(sourceDir, destinationDir, partition=1.0):

    if partition < 0.0 or partition > 1.0:
        print("Error: partition must be between 0 and 1")
        return

    # Create the destination directory if it doesn't exist
    os.makedirs(destinationDir, exist_ok=True)

    # List all files in the source directory
    ImageFiles = os.listdir(sourceDir)

    # Calculate the number of images to process (first 80%)
    imageCount = int(len(ImageFiles) * partition)

    for i, imageFile in enumerate(ImageFiles[:imageCount]):
        if imageFile.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Load the image
            imagePath = os.path.join(sourceDir, imageFile)
            image = cv2.imread(imagePath)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Use canny for edge detection
            edges = cv2.Canny(blurred, threshold1=10, threshold2=40)

            # Apply morphological to help connect stuff
            kernel = np.ones((100, 100), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create mask
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

            # Save the normalized image
            writePath = os.path.join(destinationDir, imageFile)
            cv2.imwrite(writePath, mask)
            print(f"Processed image {i + 1}/{imageCount}")

    print("Normalization and saving completed.")

def maskImage(image):
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

    return mask