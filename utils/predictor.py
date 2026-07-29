import numpy as np

def predict_disease(model, img, class_names):
    """
    Predict disease and confidence.
    """

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction) * 100)

    return predicted_class, confidence