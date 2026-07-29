from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

from utils.image_processing import preprocess_image
from utils.predictor import predict_disease


app = FastAPI(
    title="Mango Disease Prediction API",
    description="AI-based Mango Leaf Disease Detection API",
    version="1.0"
)


# -----------------------------
# CORS (For Lovable Frontend)
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Later we can restrict to Lovable URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Download & Load Model
# -----------------------------

MODEL_PATH = hf_hub_download(
    repo_id="harshalroy16/mango-disease-model",
    filename="mango_disease_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


# -----------------------------
# Disease Classes
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
# Home Route
# -----------------------------

@app.get("/")
def home():
    return {
        "status": "API Running",
        "message": "Mango Disease Prediction API"
    }


# -----------------------------
# Prediction Route
# -----------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        image = Image.open(file.file).convert("RGB")

        img = preprocess_image(image)

        disease, confidence = predict_disease(
            model,
            img,
            class_names
        )

        return {
            "disease": disease,
            "confidence": round(float(confidence), 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )