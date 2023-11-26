import cv2
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load the pre-trained model
model = keras.models.load_model('side_detection_model_v13.h5')

# Load your input image
# imageName = '/home/derrick/data/speedplusv2/synthetic/images/img031724.jpg'
imgNumber = 31724

# Define the number of rows and columns for section division
labelsList = ['none', 'Side1', 'Side2', 'Side3', 'Side4', 'Side5', 'Side6']
numClasses = len(labelsList)
squareSize = 150
modelInputSize = model.input_shape[1]
imageHeight = 1200*modelInputSize//squareSize
imageWidth = 1920*modelInputSize//squareSize
overlap = 0.05
shiftBox = int(modelInputSize * overlap)
threshold = .95

# The colormap for each class
classColors = {
    'none': (0, 0, 0),       # Black for 'none'
    'Side1': (255, 0, 0),    # Red
    'Side2': (0, 255, 0),    # Green
    'Side3': (0, 0, 255),    # Blue
    'Side4': (0, 255, 255),  # Yellow
    'Side5': (255, 0, 255),  # Magenta
    'Side6': (255, 255, 0),  # Cyan
}

# Loop through and add the next 15 images
imageList = []
for i in range(1):
    imageNum = i + imgNumber
    imageName = '/home/derrick/data/speedplusv2/synthetic/images/img0' + str(imageNum) + '.jpg'
    imageList.append(imageName)

for imageName in imageList:
    # Load the image
    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    resultImage = cv2.imread(imageName)

    # Resize the image by squareSize to modelInputSize
    image = cv2.resize(image, (imageWidth, imageHeight))

    # Loop through the sub-sections
    imageSections = []
    startRow = 0
    startCol = 0
    while startRow + modelInputSize <= imageHeight:
        while startCol + modelInputSize <= imageWidth:
            endRow = startRow + modelInputSize
            endCol = startCol + modelInputSize

            # Crop the section from the image
            sectionImage = image[startRow:endRow, startCol:endCol]

            # Add the section image to the complete image list
            imageSections.append(sectionImage)
            startCol += shiftBox
        startCol = 0
        startRow += shiftBox

    imageSections = np.array(imageSections)
    imageSections = np.expand_dims(imageSections, axis=0)
    imageSections = imageSections.reshape((-1,) + imageSections.shape[2:])
    print("Model Input Shape:", model.input_shape)
    print("imageSections Shape:", imageSections.shape)

    predictions = model.predict(imageSections)
    print("predictions Shape:", predictions.shape)

    # Loop through the sub-sections
    index = 0
    startRow = 0
    startCol = 0
    resultImage = np.zeros((imageHeight, imageWidth, 7), dtype=np.uint8)
    while startRow + modelInputSize <= imageHeight:
        while startCol + modelInputSize <= imageWidth:
            endRow = startRow + modelInputSize
            endCol = startCol + modelInputSize

            # Get the predictions for this section
            predictionsSection = predictions[index]

            #create 7 layers of predictions
            predictionsLayered = np.zeros((modelInputSize, modelInputSize, 7), dtype=np.uint8)
            for i in range(numClasses):
                predictionsLayered[:, :, i] = (predictionsSection[i] * 255).astype(np.uint8)

            resultImage[startRow:endRow, startCol:endCol] = np.maximum(resultImage[startRow:endRow, startCol:endCol], predictionsLayered)
            
            startCol += shiftBox
            index += 1
        startCol = 0
        startRow += shiftBox
    
    # Image mask from CNN
    smallImg = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow('Original', smallImg)
    for i in range(numClasses):
        layerImage = resultImage[:, :, i]
        layerImage = cv2.resize(layerImage, (0, 0), fx=0.25, fy=0.25)

        # Display the image using OpenCV
        cv2.imshow('Side{} Mask'.format(i), layerImage)
    # Initialize an empty color image
    colorImage = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)

    # Assign colors based on the class with the maximum value
    max_indices = np.argmax(resultImage, axis=2)
    for i in range(numClasses):
        if i == 0:
            classKey = 'none'
        else:
            classKey = 'Side' + str(i)
        classColor = classColors[classKey]

        # Create a mask for pixels where the class index is i
        mask = (max_indices == i)[:, :, None]

        # Assign the corresponding color to those pixels
        colorImage = np.where(mask, classColor, colorImage)

    # Display the color mask
    colorImage = colorImage.astype(np.uint8)
    cv2.imshow('Color Mask', colorImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Feature Importance View
    for i in range(numClasses):
        channel = resultImage[:, :, i]
        normalized_channel = channel / 255.0
        resizedChannel = cv2.resize(normalized_channel, (image.shape[1], image.shape[0]))

        # Convert the original image to the same data type as resizedChannel
        imageFloat = image.astype(np.float32) / 255.0

        # Multiply the original image with the resized and normalized channel
        multipliedImage = cv2.multiply(imageFloat, resizedChannel, dtype=cv2.CV_32F)

        # Apply gamma correction for contrast enhancement
        # multipliedImage = np.power(multipliedImage, 0.75)

        # Normalize the result to the range [0, 255]
        multipliedImage = np.clip(multipliedImage * 255, 0, 255).astype(np.uint8)

        # Resize the result for display
        multipliedImage = cv2.resize(multipliedImage, (0, 0), fx=0.25, fy=0.25)

        cv2.imshow('Side{} Considerations'.format(i), multipliedImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

        

    # Close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
