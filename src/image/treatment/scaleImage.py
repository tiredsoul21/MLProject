import os

import cv2

# Scale all images in a directory to the range [0, 1]
def scaleImageDir(sourceDir, destinationDir, partition=1.0):

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

            # Normalize the image
            normImage = image.astype(float) / 255.0

            # Save the normalized image
            writePath = os.path.join(destinationDir, imageFile)
            cv2.imwrite(writePath, (normImage * 255).astype('uint8'))
            print(f"Processed image {i + 1}/{imageCount}")

    print("Normalization and saving completed.")

# Scale a single image to the range [0, 1]
def scaleImage(image):
    return image.astype(float) / 255.0