import cv2
import numpy as np
from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model('object_detection_model.h5')

# Load your input image
input_image = cv2.imread('/home/derrick/data/speedplusv2/synthetic/images/img031725.jpg')

# Define the size of smaller pieces and overlap
piece_size = (192, 120)  # Adjust as needed
overlap = 0  # Adjust as needed

#double the size of the image
input_image = cv2.resize(input_image, (0, 0), fx=1, fy=1)

# Initialize an empty image for results
result_image = np.copy(input_image)


# Iterate through the image in a grid pattern
for y in range(0, input_image.shape[0], piece_size[1] - overlap):
    for x in range(0, input_image.shape[1], piece_size[0] - overlap):
        # Crop a smaller piece from the original image
        piece = input_image[y:y + piece_size[1], x:x + piece_size[0]]

        # Resize the piece while preserving the aspect ratio
        aspect_ratio = piece.shape[1] / piece.shape[0]
        # piece = cv2.resize(piece, (640, 400))

        # Feed the smaller piece through the model for object detection
        piece = np.expand_dims(piece, axis=0)  # Add a batch dimension
        predictions = model.predict(piece)
        print(predictions)
        # Check if the prediction is 1 (indicating an object)
        if predictions[0][0] > 0.75:
            # Mark the piece as containing an object
            color = (0, 255, 0)  # You can choose a color
            thickness = 2  # You can adjust the thickness
            result_image[y:y+piece_size[1], x:x+piece_size[0]] = color

# Display the result image with objects marked
# Resize image by 50%
result_image = cv2.resize(result_image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Objects Detected', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()