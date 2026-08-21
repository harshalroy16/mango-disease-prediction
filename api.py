from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from fastapi.staticfiles import StaticFiles
import os

from utils.image_processing import preprocess_image


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Mango Disease Prediction API")


# -----------------------------
# Serve Static Files
# -----------------------------
if os.path.exists("results"):
    app.mount(
        "/results",
        StaticFiles(directory="results"),
        name="results"
    )


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
# IMPORTANT: Must match training
# order exactly
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
# TFLite Model
# -----------------------------
interpreter = None


def load_model():
    global interpreter

    if interpreter is None:

        print("Downloading TFLite model...")

        model_path = hf_hub_download(
            repo_id="harshalroy16/mango-disease-model",
            filename="mango_disease_model.tflite",
            force_download=True
        )

        print("Loading TFLite model...")

        interpreter = tf.lite.Interpreter(
            model_path=model_path
        )

        interpreter.allocate_tensors()

        # Print model information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("====================================")
        print("TFLite Model Loaded")
        print("INPUT DETAILS:")
        print(input_details)
        print("OUTPUT DETAILS:")
        print(output_details)
        print("====================================")

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
async def predict(
    file: UploadFile = File(...)
):

    try:

        # -----------------------------
        # Load model
        # -----------------------------
        interpreter = load_model()

        # -----------------------------
        # Read uploaded image
        # -----------------------------
        image = Image.open(
            file.file
        ).convert("RGB")

        # -----------------------------
        # Preprocess image
        # -----------------------------
        img = preprocess_image(image)

        # -----------------------------
        # Get TFLite details
        # -----------------------------
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_info = input_details[0]

        input_index = input_info["index"]
        input_dtype = input_info["dtype"]
        input_shape = input_info["shape"]

        print("Input shape:", input_shape)
        print("Input dtype:", input_dtype)

        # -----------------------------
        # Make sure shape is correct
        # -----------------------------
        img = np.asarray(img)

        if tuple(img.shape) != tuple(input_shape):

            img = np.resize(
                img,
                input_shape
            )

        # -----------------------------
        # Handle input dtype
        # -----------------------------
        if input_dtype == np.float32:

            img = img.astype(
                np.float32
            )

        elif input_dtype == np.uint8:

            img = np.clip(
                img,
                0,
                255
            ).astype(
                np.uint8
            )

        elif input_dtype == np.int8:

            img = np.clip(
                img,
                -128,
                127
            ).astype(
                np.int8
            )

        else:

            raise ValueError(
                f"Unsupported TFLite input dtype: {input_dtype}"
            )

        # -----------------------------
        # Set model input
        # -----------------------------
        interpreter.set_tensor(
            input_index,
            img
        )

        # -----------------------------
        # Run inference
        # -----------------------------
        interpreter.invoke()

        # -----------------------------
        # Get prediction
        # -----------------------------
        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        # -----------------------------
        # Convert output if necessary
        # -----------------------------
        prediction = np.asarray(
            prediction,
            dtype=np.float32
        )

        # If output isn't already
        # probability-like values,
        # apply softmax.
        if (
            np.min(prediction) < 0
            or np.max(prediction) > 1
            or not np.isclose(
                np.sum(prediction),
                1.0,
                atol=0.01
            )
        ):

            exp_prediction = np.exp(
                prediction - np.max(prediction)
            )

            prediction = (
                exp_prediction /
                np.sum(exp_prediction)
            )

        # -----------------------------
        # Prediction
        # -----------------------------
        predicted_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[predicted_index] * 100
        )

        predicted_class = class_names[
            predicted_index
        ]

        # -----------------------------
        # Print prediction details
        # -----------------------------
        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            confidence
        )

        print(
            "All probabilities:",
            prediction
        )

        # -----------------------------
        # Response
        # -----------------------------
        return {

            "success": True,

            "disease": predicted_class,

            "confidence": round(
                confidence,
                2
            )
        }

    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return {

            "success": False,

            "error": str(e)
        }