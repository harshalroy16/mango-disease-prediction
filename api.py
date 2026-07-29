from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import os

from utils.image_processing import preprocess_image
from utils.predictor import predict_disease

from huggingface_hub import hf_hub_download


app = FastAPI(title="Mango Disease Prediction API")


model = None


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
    global model

    if model is None:
        print("Loading model...")

        model_path = hf_hub_download(
            repo_id="harshalroy16/mango-disease-model",
            filename="mango_disease_model.keras"
        )

        model = tf.keras.models.load_model(model_path)

        print("Model loaded successfully")

    return model



@app.get("/")
def home():
    return {
        "status": "API Running",
        "message": "Mango Disease Prediction API"
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    model = load_model()

    image = Image.open(file.file).convert("RGB")

    img = preprocess_image(image)

    disease, confidence = predict_disease(
        model,
        img,
        class_names
    )

    return {
        "disease": disease,
        "confidence": round(float(confidence),2)
    }