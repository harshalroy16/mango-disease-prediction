import numpy as np
import cv2

def preprocess_image(image):
    """
    Preprocess image for the trained model.
    """

    image = np.array(image)

    # Resize
    image = cv2.resize(image, (224, 224))

    # Convert to float32 and normalize
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image