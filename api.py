from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from huggingface_hub import hf_hub_download
from utils.image_processing import preprocess_image


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Mango Disease Prediction API")
# Serve static files
app.mount("/results", StaticFiles(directory="results"), name="results")


# -----------------------------
# CORS Configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Model Classes
# -----------------------------
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


# -----------------------------
# Load TFLite Model
# -----------------------------
interpreter = None


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


# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "API Running",
        "message": "Mango Disease Prediction API"
    }


# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        interpreter = load_model()

        # Read image
        image = Image.open(file.file).convert("RGB")

        # Preprocess image
        img = preprocess_image(image).astype(np.float32)

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input
        interpreter.set_tensor(
            input_details[0]["index"],
            img
        )

        # Run prediction
        interpreter.invoke()

        # Get output
        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        return {
            "success": True,
            "disease": class_names[predicted_index],
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        print("Prediction Error:", str(e))

        return {
            "success": False,
            "error": str(e)
        }