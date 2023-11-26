import os
import sys
import json
import argparse
import numpy as np
import cv2
from random import shuffle, seed
from tensorflow import keras
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

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
    maxNegatives = 12000
    numClasses = len(labelsList)
    imageHeight = 1200 
    imageWidth = 1920
    seedValue = 42
    squareSize = 256
    overlap = 0.30
    shiftBox = int(squareSize * overlap)
    # Define the threshold for the number of mask pixels
    threshold = squareSize*squareSize*0.25

    # Initialize the complete label list
    labelSections = []

    # Randomize the order of the images
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

        # Loop through the sub-sections
        startRow = 0
        startCol = 0
        while startRow + squareSize <= imageHeight:
            while startCol + squareSize <= imageWidth:
                endRow = startRow + squareSize
                endCol = startCol + squareSize

                sectionLabel = "none"
                channelCounts = np.count_nonzero(masks[startRow:endRow, startCol:endCol, :], axis = (0,1))

                # Check if any channel count is above the threshold
                if any(count > threshold for count in channelCounts):
                    # Find the index of the channel with the maximum count
                    max_channel = np.argmax(channelCounts)
                    sectionLabel = classIDs[max_channel]

                # Add the section label to the complete label list
                labelSections.append(sectionLabel)
                startCol += shiftBox
            startCol = 0
            startRow += shiftBox
                
    print(f"Label creation complete {len(labelSections)} labels created.")

    # Convert labels to one-hot encoding
    oneHotLabels = []
    for label in labelSections:
        oneHot = [0] * len(labelsList)
        index = labelsList.index(label)
        oneHot[index] = 1
        oneHotLabels.append(oneHot)

    imageSections = []
    # Loop through the images and create labels for sub-sections
    for imageFilename in keyList:
        imageInfo = labels[imageFilename]

        # Load the image
        image = cv2.imread(os.path.join(dataPath, imageFilename), cv2.IMREAD_GRAYSCALE)

        # Loop through the sub-sections
        startRow = 0
        startCol = 0
        while startRow + squareSize <= imageHeight:
            while startCol + squareSize <= imageWidth:
                endRow = startRow + squareSize
                endCol = startCol + squareSize

                # Crop the section from the image
                sectionImage = image[startRow:endRow, startCol:endCol]

                # Add the section image to the complete image list
                imageSections.append(sectionImage)
                startCol += shiftBox
            startCol = 0
            startRow += shiftBox
                
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
    model.add(layers.Conv2D(64, (9,9), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))
    model.add(layers.Conv2D(64, (6,6), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))
    model.add(layers.Conv2D(64, (9,9), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))
    model.add(layers.Conv2D(64, (6,6), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))
    model.add(layers.Conv2D(64, (9,9), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))
   
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (9,9), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (9,9), activation='relu', input_shape=(squareSize, squareSize, 1),padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(numClasses, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.00002), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Split the dataset into training, validation, and test sets
    trainSize = int(0.6 * len(imageSections))
    valSize = int(0.2 * len(imageSections))
    trainImages = imageSections[:trainSize]
    valImages = imageSections[trainSize:trainSize + valSize]
    testImages = imageSections[trainSize + valSize:]
    trainLabels = labels[:trainSize]
    valLabels = labels[trainSize:trainSize + valSize]
    testLabels = labels[trainSize + valSize:]

    # Remove samples and labels from the training set
    numSamplesToMove = max(0, np.sum(trainLabels[:, 0]) - maxNegatives)
    indicesRemoved = np.where(trainLabels[:, 0] == 1)[0][:numSamplesToMove]
    trainImages = np.delete(trainImages, indicesRemoved, axis=0)
    trainLabels = np.delete(trainLabels, indicesRemoved, axis=0)
    numSamplesToMove = max(0, np.sum(valLabels[:, 0]) - maxNegatives)
    indicesRemoved = np.where(valLabels[:, 0] == 1)[0][:numSamplesToMove]
    valImages = np.delete(valImages, indicesRemoved, axis=0)
    valLabels = np.delete(valLabels, indicesRemoved, axis=0)
    numSamplesToMove = max(0, np.sum(testLabels[:, 0]) - int(maxNegatives/3))
    indicesRemoved = np.where(testLabels[:, 0] == 1)[0][:numSamplesToMove]
    testImages = np.delete(testImages, indicesRemoved, axis=0)
    testLabels = np.delete(testLabels, indicesRemoved, axis=0)

    # Print the new sum of occurrences for each side in the training set
    sumPerSide = np.sum(trainLabels, axis=0)
    print("Sum of each side's occurrences in the training set:")
    print(sumPerSide)
    sumPerSide = np.sum(valLabels, axis=0)
    print("Sum of each side's occurrences in the validation set:")
    print(sumPerSide)
    sumPerSide = np.sum(testLabels, axis=0)
    print("Sum of each side's occurrences in the test set:")
    print(sumPerSide)

    # Reshape the images
    trainImages = np.expand_dims(trainImages, axis=-1)
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(trainImages)

    # Train the model
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(trainImages, trainLabels, validation_data=(valImages, valLabels), epochs=100, batch_size=16, callbacks=[earlyStopping])

    print("Training complete, Saving.")
    model.save('side_detection_model.h5')
    # model = keras.models.load_model('side_detection_model.h5')

    # Evaluate the model
    print ("Evaluating model")
    testLoss, testAccuracy = model.evaluate(testImages, testLabels)
    print(f"Test loss: {testLoss}")
    print(f"Test accuracy: {testAccuracy}")

    # Evaluate the model
    batchSize = 32
    predictions_list = []
    trueLabelsList = []

    # Iterate over the test set in batches
    for i in range(0, len(testImages), batchSize):
        batchImages = testImages[i:i + batchSize]
        batchLabels = testLabels[i:i + batchSize]

        # Perform batch-wise prediction
        batchPredictions = model.predict(batchImages)
        
        # Append predictions and true labels to lists
        predictions_list.append(batchPredictions)
        trueLabelsList.append(batchLabels)

    # Concatenate predictions and true labels from all batches
    predictions = np.concatenate(predictions_list, axis=0)
    trueLabels = np.concatenate(trueLabelsList, axis=0)

    # Process predictions and true labels
    predicted_labels = [round(prediction.argmax()) for prediction in predictions]
    trueLabels = [label.argmax() for label in trueLabels]

    # Calculate confusion matrix and classification report
    confMatrix = confusion_matrix(trueLabels, predicted_labels)
    classReport = classification_report(trueLabels, predicted_labels)

    # Print the results
    print("Confusion Matrix:")
    print(confMatrix)
    print("Classification Report:")
    print(classReport)

if __name__ == "__main__":
    main()