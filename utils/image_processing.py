import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess uploaded PIL image for prediction.
    Returns a normalized image ready for the model.
    """

    # PIL -> NumPy
    image_cv = np.array(image)

    # RGB -> BGR
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Resize
    image_cv = cv2.resize(image_cv, (224, 224))

    # Noise Removal
    image_cv = cv2.GaussianBlur(image_cv, (3, 3), 0)

    # Contrast Enhancement
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    image_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # BGR -> RGB
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Normalize
    image_cv = image_cv.astype(np.float32) / 255.0

    # Add batch dimension
    image_cv = np.expand_dims(image_cv, axis=0)

    return image_cv