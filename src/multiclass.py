import os
import sys
import json
import argparse
import numpy as np
import cv2

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
    numClasses = len(labelsList)
    imageHeight = 1200 
    imageWidth = 1920
    num_rows = 24
    num_cols = 24
    threshold = 30


    subHeight = imageHeight // num_rows
    subWidth = imageWidth // num_cols
    # Define the threshold for the number of mask pixels

    # Initialize the complete label list
    labelSections = []

    # Loop through the images and create labels for sub-sections
    for imageFilename in list(labels.keys()):
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
        for row in range(num_rows):
            for col in range(num_cols):
                # Define the section boundaries
                startRow = row * subHeight
                endRow = (row + 1) * subHeight
                startCol = col * subWidth
                endCol = (col + 1) * subWidth

                sectionLabels = []

                # Add label for sub-section with channel > threshold
                for i in range(3):
                    if np.sum(masks[startRow:endRow, startCol:endCol, i]) > threshold:
                        sectionLabels.append(classIDs[i])
                if len(sectionLabels) == 0:
                    sectionLabels.append("none")

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
    for imageFilename in list(labels.keys()):
        imageInfo = labels[imageFilename]

        # Load the image
        image = cv2.imread(os.path.join(dataPath, imageFilename))

        # Loop through the sub-sections
        for row in range(num_rows):
            for col in range(num_cols):
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

    # Define the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(subHeight, subWidth, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(numClasses, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Split the dataset into training, validation, and test sets
    trainSize = int(0.6 * len(imageSections))
    valSize = int(0.2 * len(imageSections))
    trainImages = imageSections[:trainSize]
    valImages = imageSections[trainSize:trainSize + valSize]
    testImages = imageSections[trainSize + valSize:]
    trainLabels = labels[:trainSize]
    valLabels = labels[trainSize:trainSize + valSize]
    testLabels = labels[trainSize + valSize:]

    # Train the model
    model.fit(trainImages, trainLabels, validation_data=(valImages, valLabels), epochs=10, batch_size=16)

    # Evaluate the model
    testLoss, testAccuracy = model.evaluate(testImages, testLabels)

    print(f"Test loss: {testLoss}")
    print(f"Test accuracy: {testAccuracy}")

    # Save the trained model for future use
    model.save('side_detection_model.h5')

if __name__ == "__main__":
    main()
