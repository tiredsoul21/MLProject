import os
import sys
import json
import argparse
import numpy as np
import cv2
# import tensorflow as tf
from tensorflow.keras import models, layers
from image.treatment.scaleImage import scaleImageDir
from image.treatment.maskImage import maskImageDir
# tf.config.threading.set_inter_op_parallelism_threads(16)
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

    dataPath = config.get("dataPath")
    if dataPath is None:
        print("Error: Missing required 'dataPath' in config")
        return
    print(f"Using data file: {dataPath}")

    # Read in the data file
    droplocation1 = dataPath + "/../scaled"
    # scaleImageDir(dataPath, droplocation1)
    droplocation2 = dataPath + "/../masked"
    # maskImageDir(droplocation1, droplocation2)

    # Define the number of rows and columns for section division
    num_rows = 10
    num_cols = 10

    # Define the threshold for the number of mask pixels
    threshold = 200

    # Initialize lists to store image sections and their corresponding labels
    image_sections = []
    labels = []
    
    # Loop through the images in droplocation1 and their corresponding masks in droplocation2
    for imageFile in os.listdir(droplocation2):
        if imageFile.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Load the image
            imagePath = os.path.join(droplocation1, imageFile)
            image = cv2.imread(imagePath)

            # Load the corresponding mask
            maskPath = os.path.join(droplocation2, imageFile)
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
            # mask = cv2.resize(mask, (0, 0), fx=2, fy=2)

            # Calculate the section size
            section_height = image.shape[0] // num_rows
            section_width = image.shape[1] // num_cols

            for row in range(num_rows):
                for col in range(num_cols):
                    # Define the section boundaries
                    start_row = row * section_height
                    end_row = (row + 1) * section_height
                    start_col = col * section_width
                    end_col = (col + 1) * section_width

                    # Crop the section from the image and mask
                    section_image = image[start_row:end_row, start_col:end_col]
                    section_mask = mask[start_row:end_row, start_col:end_col]

                    # Calculate the number of mask pixels in the section
                    num_mask_pixels = np.count_nonzero(section_mask)

                    # Determine the label based on the threshold
                    label = 1 if num_mask_pixels >= threshold else 0

                    # Append the section image and label to their respective lists
                    image_sections.append(section_image)
                    labels.append(label)

    cv2.destroyAllWindows()  # Close the OpenCV window when done

    # Convert the lists to numpy arrays
    image_sections = np.array(image_sections)
    labels = np.array(labels)

    # Define the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(section_height, section_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Split your dataset into training, validation, and test sets
    train_size = int(0.6 * len(image_sections))
    val_size = int(0.2 * len(image_sections))
    train_images = image_sections[:train_size]
    train_labels = labels[:train_size]
    val_images = image_sections[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    test_images = image_sections[train_size + val_size:]
    test_labels = labels[train_size + val_size:]

    # Train the model
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, batch_size=16)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    # Save the trained model for future use
    model.save('object_detection_model.h5')

if __name__ == "__main__":
    main()
