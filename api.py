from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf

from utils.image_processing import preprocess_image
from utils.predictor import predict_disease

app = FastAPI(title="Mango Disease Prediction API")

# Load model only once
model = tf.keras.models.load_model("models/mango_disease_model.keras")

# Disease classes
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


@app.get("/")
def home():
    return {
        "status": "API Running",
        "message": "Mango Disease Prediction API"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    img = preprocess_image(image)

    disease, confidence = predict_disease(
        model,
        img,
        class_names
    )

    return {
        "disease": disease,
        "confidence": round(confidence, 2)
    }