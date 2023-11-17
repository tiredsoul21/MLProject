import os
import sys
import json
import argparse
import numpy as np
import cv2
from random import shuffle, seed
from keras.optimizers import Adam
from tensorflow.keras import models, layers

# Read in the CLI arguments
def parseArgs():
    configPath = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == '-c':
            if i + 1 < len(sys.argv):
                configPath = sys.argv[i + 1]
                i += 2  # Skip the value
            else:
                print("Error: Missing value for -c option")
                return None
        else:
            print(f"Error: Unknown option: {arg}")
            return None
        i += 1

    if configPath == None:
        print("Error: Missing -c option")
        return None

    return configPath

# Main function
def main():
    # Get the command line arguments
    configPath = parseArgs()
    if configPath is None:
        return
    print(f"Using config file: {configPath}")

    # Read in the Config file (JSON)
    file = open(configPath, 'r')
    config = json.load(file)
    file.close()

    # Load the data path
    dataPath = config.get("dataPath")
    if dataPath is None:
        print("Error: Missing required 'dataPath' in config")
        return
    print(f"Using data file: {dataPath}")

    # Load the labels path
    labelsPath = config.get("labels")
    if labelsPath is None:
        print("Error: Missing required 'labels' in config")
        return
    print(f"Using labels: {labelsPath}")

    # Load the labels file
    with open(labelsPath, 'r') as file:
        labels = json.load(file)

    # Define the number of rows and columns for section division
    labelsList = ['none', 'Side1', 'Side2', 'Side3', 'Side4', 'Side5', 'Side6']
    maxNegatives = 60000
    numClasses = len(labelsList)
    imageHeight = 1200 
    imageWidth = 1920
    numRows = 25
    numCols = 40
    seedValue = 42


    subHeight = imageHeight // numRows
    subWidth = imageWidth // numCols
    # Define the threshold for the number of mask pixels
    threshold = subHeight*subWidth*0.25


    # Initialize the complete label list
    labelSections = []

    seed(seedValue)
    keyList = list(labels.keys())
    shuffle(keyList)

    # Loop through the images and create labels for sub-sections
    for imageFilename in keyList:
        imageInfo = labels[imageFilename]

        #create a poly array
        poly = []
        classIDs = []
        for region_id, region_info in imageInfo.get("regions", {}).items():
            # Get the polygon coordinates
            xPoints = region_info["shape_attributes"]["all_points_x"]
            yPoints = region_info["shape_attributes"]["all_points_y"]

            # Create a polygon and add it to the list
            polygon = np.array(list(zip(xPoints, yPoints)), np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            poly.append(polygon)
            classIDs.append(region_info["region_attributes"]["label"])

        # If poly has more than 3 polygons the image is not valid
        if len(poly) > 3:
            print(f"Error: Image {imageFilename} has more than 3 polygons")
            continue

        # Create the masks array
        masks = []
        # Create the masks with polyfill -- add in labels
        for i in range(len(poly)):
            mask = np.zeros((1200, 1920), np.uint8)
            cv2.fillPoly(mask, [poly[i]], 255)
            masks.append(mask)
        # Add in missing masks -- if there are less than 3 polygons
        for i in range(3 - len(poly)):
            masks.append(np.zeros((1200, 1920), np.uint8))
        # Create 3 channels for the masks
        masks = np.array(masks)
        masks = np.moveaxis(masks, 0, -1)

        # # Display the masks
        # cv2.imshow("Masks", masks)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Loop through the sub-sections
        for row in range(numRows):
            for col in range(numCols):
                # Define the section boundaries
                startRow = row * subHeight
                endRow = (row + 1) * subHeight
                startCol = col * subWidth
                endCol = (col + 1) * subWidth

                sectionLabels = []
                channelCounts = np.count_nonzero(masks[startRow:endRow, startCol:endCol, :], axis = (0,1))

                # Add label for sub-section with channel > threshold
                for i in range(3):
                    # print(classIDs[i])
                    # print(channelCounts[i])
                    if channelCounts[i] > threshold:
                        sectionLabels.append(classIDs[i])
                if len(sectionLabels) == 0:
                    sectionLabels.append("none")

                #display sub image
                # cv2.imshow("Masks", masks[startRow:endRow, startCol:endCol, :])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Add the section labels to the complete label list
                labelSections.append(sectionLabels)
                
    print(f"Label creation complete {len(labelSections)} labels created.")
    # print("Image Labels:")
    # print(labelSections)

    #Convert labels to one-hot encoding
    oneHotLabels = []
    for label in labelSections:
        oneHot = [0] * len(labelsList)
        for i in range(len(label)):
            oneHot[labelsList.index(label[i])] = 1
        oneHotLabels.append(oneHot)

    # print("One Hot Labels:")
    # print(oneHotLabels)

    imageSections = []
    # Loop through the images and create labels for sub-sections
    for imageFilename in keyList:
        imageInfo = labels[imageFilename]

        # Load the image
        image = cv2.imread(os.path.join(dataPath, imageFilename), cv2.IMREAD_GRAYSCALE)

        # Loop through the sub-sections
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
                
    print(f"Image treatment complete. {len(imageSections)} images created.")

    # Convert the lists to numpy arrays
    imageSections = np.array(imageSections)
    labels = np.array(oneHotLabels)

    # Print the result
    sumPerSide = np.sum(labels, axis=0)
    print("Sum of each side's occurrences:")
    print(sumPerSide)

    # Define the model
    model = models.Sequential()
    # model.add(layers.Conv2D(4, (3,3), activation='relu', input_shape=(subHeight, subWidth, 1),padding='same'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(subHeight, subWidth, 1),padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(subHeight, subWidth, 1),padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(subHeight, subWidth, 1),padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(numClasses, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Split the dataset into training, validation, and test sets
    trainSize = int(0.6 * len(imageSections))
    valSize = int(0.2 * len(imageSections))
    trainImages = imageSections[:trainSize]
    valImages = imageSections[trainSize:trainSize + valSize]
    testImages = imageSections[trainSize + valSize:]
    trainLabels = labels[:trainSize]
    valLabels = labels[trainSize:trainSize + valSize]
    testLabels = labels[trainSize + valSize:]

    # Calculate the number of samples to move to the test set for the first class
    sumPerSide = np.sum(trainLabels, axis=0)
    secondMax = np.argsort(-sumPerSide)[1]
    numSamplesToMove = max(0, np.sum(trainLabels[:, 0]) - maxNegatives)

    # Identify the indices of the samples to move
    indices_to_move = np.where(trainLabels[:, 0] == 1)[0][:numSamplesToMove]

    # Move the selected samples and labels to the test set
    # testImages = np.concatenate([testImages, trainImages[indices_to_move]])
    # testLabels = np.concatenate([testLabels, trainLabels[indices_to_move]])
    # Remove the moved samples and labels from the training set
    trainImages = np.delete(trainImages, indices_to_move, axis=0)
    trainLabels = np.delete(trainLabels, indices_to_move, axis=0)

    # Print the new sum of occurrences for each side in the training set
    sumPerSide = np.sum(trainLabels, axis=0)
    print("Sum of each side's occurrences in the training set:")
    print(sumPerSide)

    # Print the new sum of occurrences for each side in the test set
    sumPerSide = np.sum(valLabels, axis=0)
    print("Sum of each side's occurrences in the validation set:")
    print(sumPerSide)

    # Print the new sum of occurrences for each side in the test set
    sumPerSide = np.sum(testLabels, axis=0)
    print("Sum of each side's occurrences in the test set:")
    print(sumPerSide)

    # Train the model
    model.fit(trainImages, trainLabels, validation_data=(valImages, valLabels), epochs=3, batch_size=16)

    # Evaluate the model
    testLoss, testAccuracy = model.evaluate(testImages, testLabels)

    print(f"Test loss: {testLoss}")
    print(f"Test accuracy: {testAccuracy}")

    # Save the trained model for future use
    model.save('side_detection_model.h5')

if __name__ == "__main__":
    main()
