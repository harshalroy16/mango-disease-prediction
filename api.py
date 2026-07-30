from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np

from huggingface_hub import hf_hub_download

from utils.image_processing import preprocess_image

app = FastAPI(title="Mango Disease Prediction API")

interpreter = None

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


def load_model():
    global interpreter

    if interpreter is None:
        print("Downloading TFLite model...")

        model_path = hf_hub_download(
            repo_id="harshalroy16/mango-disease-model",
            filename="mango_disease_model.tflite"
        )

        print("Loading interpreter...")

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        print("TFLite model loaded.")

    return interpreter


@app.get("/")
def home():
    return {
        "status": "API Running",
        "message": "Mango Disease Prediction API"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    interpreter = load_model()

    image = Image.open(file.file).convert("RGB")

    img = preprocess_image(image).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"],
        img
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    predicted_index = int(np.argmax(prediction))

    confidence = float(np.max(prediction) * 100)

    return {
        "disease": class_names[predicted_index],
        "confidence": round(confidence, 2)
    }