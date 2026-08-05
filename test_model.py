import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("models/mango_disease_model.keras")

class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# CHANGE THIS PATH
image_path = "test.jpg"

img = Image.open(image_path).convert("RGB")
img = img.resize((224,224))

img = np.array(img).astype(np.float32)

img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

print(prediction)

print("Predicted Class:", class_names[np.argmax(prediction)])

print("Confidence:", np.max(prediction)*100)