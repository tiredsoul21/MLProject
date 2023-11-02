import os
import cv2
import numpy as np

# Function to handle mouse events
def selectRegion(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(param['selection']) < 2:
            param['selection'].append((x, y))
        else:
            param['selection'] = [(x, y)]  # Start a new selection if two points are already selected

    # Draw the rectangle on the image as the user selects the region
    if len(param['selection']) == 2:
        cv2.rectangle(param['image'], param['selection'][0], param['selection'][1], (0, 255, 0), 2)
    
    cv2.imshow('Image', param['image'])


# Function to mask the selected region
def maskSelectedRegion(image, selection):
    mask = np.zeros_like(image)
    if len(selection) == 2:
        cv2.rectangle(mask, selection[0], selection[1], (255, 255, 255), thickness=cv2.FILLED)
    return mask

# Main function to process a single image

def processSingleImage(image):
    # Resize the image to 50% of its original size
    resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    selection = []
    cv2.imshow('Image', resized_image)
    cv2.setMouseCallback('Image', selectRegion, {'image': resized_image, 'selection': selection})

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            # print("Creating mask...")
            if len(selection) == 2:
                mask = maskSelectedRegion(resized_image, selection)
                # print("Mask created.")
                break
        # elif key == ord('q'):
            # print("Quitting...")
            # break

    # cv2.destroyAllWindows()
    return mask

# Scale all images in a directory to the range [0, 1] and save masks
def maskImageDir(sourceDir, destinationDir, partition=1.0):
    if partition < 0.0 or partition > 1.0:
        print("Error: partition must be between 0 and 1")
        return

    os.makedirs(destinationDir, exist_ok=True)
    ImageFiles = os.listdir(sourceDir)
    imageCount = int(len(ImageFiles) * partition)

    localCount = 0

    for i, imageFile in enumerate(ImageFiles[:imageCount]):
        if imageFile.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            imagePath = os.path.join(sourceDir, imageFile)
            image = cv2.imread(imagePath)
            mask = processSingleImage(image)


            #save the image
            cv2.imwrite(os.path.join(destinationDir, imageFile), mask)
            localCount += 1
            print("localCount: ", localCount)
            
            # After processing an image, wait for 'q' key to move to the next image
            # key = cv2.waitKey(0) & 0xFF
            # if key == ord('q'):
            #     continue

    print("Normalization and saving completed.")
