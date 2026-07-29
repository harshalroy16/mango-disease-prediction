import os
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.image_processing import preprocess_image
from utils.predictor import predict_disease
from utils.history import save_prediction, load_history
from utils.pdf_report import generate_pdf


from disease_info import disease_info


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mango Disease Prediction",
    page_icon="🥭",
    layout="centered"
)


# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f7fff7;
}

h1 {
    color: #2e7d32;
    text-align: center;
}

h2, h3 {
    color: #388e3c;
}

.stButton button {
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}

[data-testid="stMetricValue"] {
    color: #2e7d32;
    font-size: 28px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/mango_disease_model.keras"
    )


model = load_model()


# -----------------------------
# Class Names
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
# Sidebar
# -----------------------------
st.sidebar.title("🥭 Mango Disease Prediction")

st.sidebar.info("""
AI-based mango leaf disease detection system.

Technologies:

- Python
- TensorFlow/Keras
- CNN
- OpenCV
- Streamlit
- Pandas
- Matplotlib
""")


# -----------------------------
# Title
# -----------------------------
st.title("🥭 AI Mango Disease Prediction System")

st.write(
    "Upload a mango leaf image and the AI model will predict the disease."
)


# -----------------------------
# Dashboard
# -----------------------------
history = load_history()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📊 Total Predictions",
        len(history)
    )


with col2:
    healthy_count = len(
        history[history["Disease"] == "Healthy"]
    )

    st.metric(
        "🌿 Healthy",
        healthy_count
    )


with col3:
    diseased_count = len(history) - healthy_count

    st.metric(
        "🦠 Diseased",
        diseased_count
    )


with col4:

    if len(history) > 0:
        most_common = history["Disease"].mode()[0]
    else:
        most_common = "-"

    st.metric(
        "🏆 Most Common",
        most_common
    )


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a mango leaf image",
    type=["jpg", "jpeg", "png"]
)



# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")


    st.image(
        image,
        caption="Uploaded Mango Leaf",
        width=350
    )


    # -----------------------------
    # Save Uploaded Image
    # -----------------------------
    if not os.path.exists("uploads"):
        os.makedirs("uploads")


    image_path = f"uploads/{uploaded_file.name}"

    image.save(image_path)



    # -----------------------------
    # Image Processing
    # -----------------------------
    img = preprocess_image(image)



    # -----------------------------
    # Prediction
    # -----------------------------
    predicted_class, confidence = predict_disease(
        model,
        img,
        class_names
    )



    # -----------------------------
    # Result
    # -----------------------------
    st.success(
        f"🥭 Predicted Disease: {predicted_class}"
    )


    st.progress(
        confidence / 100
    )


    st.info(
        f"🎯 Confidence: {confidence:.2f}%"
    )



    # -----------------------------
    # Save History
    # -----------------------------
    save_prediction(
        predicted_class,
        confidence
    )



    # -----------------------------
    # Confidence Chart
    # -----------------------------
    st.subheader(
        "📊 Prediction Confidence"
    )


    fig, ax = plt.subplots(
        figsize=(5, 3)
    )


    ax.bar(
        ["Confidence"],
        [confidence]
    )


    ax.set_ylim(
        0,
        100
    )


    ax.set_ylabel(
        "Percentage"
    )


    st.pyplot(fig)



    # -----------------------------
    # Disease Information
    # -----------------------------
    if predicted_class in disease_info:


        st.subheader(
            "📖 Disease Description"
        )

        st.write(
            disease_info[predicted_class]["description"]
        )


        st.subheader(
            "💊 Treatment"
        )

        st.write(
            disease_info[predicted_class]["treatment"]
        )


        st.subheader(
            "🌱 Prevention"
        )

        st.write(
            disease_info[predicted_class]["prevention"]
        )



        # -----------------------------
        # PDF Report
        # -----------------------------
        pdf_file = "Mango_Disease_Report.pdf"


        generate_pdf(
            pdf_file,
            predicted_class,
            confidence,
            disease_info[predicted_class]["description"],
            disease_info[predicted_class]["treatment"],
            disease_info[predicted_class]["prevention"]
        )


        with open(pdf_file, "rb") as file:

            st.download_button(
                label="📄 Download Prediction Report",
                data=file,
                file_name="Mango_Disease_Report.pdf",
                mime="application/pdf"
            )


    else:

        st.warning(
            "Disease information not available."
        )



# -----------------------------
# Prediction History
# -----------------------------
st.subheader(
    "📜 Prediction History"
)


history = load_history()


st.dataframe(
    history,
    use_container_width=True
)
# -----------------------------
# Export History CSV
# -----------------------------
if len(history) > 0:

    csv = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="mango_prediction_history.csv",
        mime="text/csv"
    )



# -----------------------------
# Disease Distribution Chart
# -----------------------------
if len(history) > 0:


    st.subheader(
        "📊 Disease Distribution"
    )


    disease_count = history["Disease"].value_counts()



    fig, ax = plt.subplots(
        figsize=(7, 4)
    )


    ax.pie(
        disease_count,
        labels=disease_count.index,
        autopct="%1.1f%%",
        startangle=90
    )


    ax.set_title(
        "Disease Distribution"
    )


    st.pyplot(fig)
    