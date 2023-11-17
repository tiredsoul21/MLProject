import cv2
import numpy as np
from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model('side_detection_model.h5')

# Load your input image
imageName = '/home/derrick/data/speedplusv2/synthetic/images/img031724.jpg'
imgNumber = 31724

# Define the number of rows and columns for section division
labelsList = ['none', 'Side1', 'Side2', 'Side3', 'Side4', 'Side5', 'Side6']
numClasses = len(labelsList)
imageHeight = 1200 
imageWidth = 1920

# calculate rows and cols and cast as int
numRows = int(imageHeight/model.input_shape[1])
numCols = int(imageWidth/model.input_shape[2])


threshold = .95

# The colormap for each class
class_colors = {
    'none': (0, 0, 0),       # Black for 'none'
    'Side1': (0, 0, 255),    # Red
    'Side2': (0, 255, 0),    # Green
    'Side3': (255, 0, 0),    # Blue
    'Side4': (255, 255, 0),  # Yellow
    'Side5': (255, 0, 255),  # Magenta
    'Side6': (0, 255, 255),  # Cyan
}
# Calculate sub-section height and width
subHeight = imageHeight // numRows
subWidth = imageWidth // numCols

# Loop through and add the next 15 images
imageList = []
for i in range(5):
    imageNum = i + imgNumber
    imageName = '/home/derrick/data/speedplusv2/synthetic/images/img0' + str(imageNum) + '.jpg'
    imageList.append(imageName)

for imageName in imageList:
    # Load the image
    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    resultImage = cv2.imread(imageName)

    # Loop through the sub-sections
    imageSections = []
    for row in range(numRows):
        for col in range(numCols):
            # Define the section boundaries
            startRow = row * subHeight
            endRow = (row + 1) * subHeight
            startCol = col * subWidth
            endCol = (col + 1) * subWidth

            # Crop the section from the image
            sectionImage = image[startRow:endRow, startCol:endCol]

            # Add the section image to the complete image list
            imageSections.append(sectionImage)

    imageSections = np.array(imageSections)
    imageSections = np.expand_dims(imageSections, axis=0)
    imageSections = imageSections.reshape((-1,) + imageSections.shape[2:])
    print("Model Input Shape:", model.input_shape)
    print("imageSections Shape:", imageSections.shape)

    predictions = model.predict(imageSections)
    print("predictions Shape:", predictions.shape)

    # Loop through the sub-sections
    index = 0
    for row in range(numRows):
        for col in range(numCols):
            # Define the section boundaries
            startRow = row * subHeight
            endRow = (row + 1) * subHeight
            startCol = col * subWidth
            endCol = (col + 1) * subWidth

            # Crop the section from the image
            sectionImage = image[startRow:endRow, startCol:endCol]

            # Reshape the input for the model
            sectionImage = np.expand_dims(sectionImage, axis=0)

            # Choose the classes based on the threshold
            detectedClasses = [labelsList[i] for i in range(numClasses) if predictions[index, i] > threshold]
            # Emphasize the detected regions in the result image
            for detected_class in detectedClasses:
                color = class_colors[detected_class]
                resultImage[startRow:endRow, startCol:endCol, :] = np.clip(
                    resultImage[startRow:endRow, startCol:endCol, :] + color, 0, 255
                )
            index += 1

    # Display the result image with objects marked
    # Resize image by 50%
    resultImage = cv2.resize(resultImage, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Objects Detected', resultImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()