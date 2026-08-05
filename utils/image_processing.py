import cv2
import numpy as np
import tensorflow as tf


def preprocess_image(image):
    """
    Preprocess image for MobileNetV2.
    """

    # PIL -> NumPy
    image = np.array(image)

    # Resize
    image = cv2.resize(image, (224, 224))

    # Convert to float32
    image = image.astype(np.float32)

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image