import numpy as np
import cv2

def preprocess_image(image):
    """
    Preprocess image for the trained model.
    The model expects raw pixel values in the 0-255 range.
    """

    # Convert PIL image to NumPy array
    image = np.array(image)

    # Resize to model input size
    image = cv2.resize(image, (224, 224))

    # Convert to float32
    # IMPORTANT: Do NOT divide by 255.0
    image = image.astype(np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image
