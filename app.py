
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# 1. Page configuration
# -------------------------------

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Image Classification")

st.write(
    "An educational deep-learning project using a custom CNN "
    "and transfer learning with MobileNetV2."
)

st.warning(
    "Research demonstration only. This application is not a "
    "medical diagnostic tool. Predictions must not be used to "
    "diagnose or rule out a brain tumor. Consult a qualified "
    "medical professional for clinical interpretation."
)

# -------------------------------
# 2. Check trained files
# -------------------------------

class_path = "models/class_names.json"

if not os.path.exists(class_path):
    st.error(
        "Class labels not found. Run train.py successfully first."
    )
    st.stop()

with open(class_path, "r") as f:
    class_names = json.load(f)


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# -------------------------------
# 3. Choose a model
# -------------------------------

model_options = {
    "Custom CNN": "models/custom_cnn.keras",
    "MobileNetV2 (Transfer Learning)": "models/mobilenetv2.keras"
}

selected_model = st.selectbox(
    "Select a trained model",
    list(model_options.keys())
)

model_path = model_options[selected_model]

if not os.path.exists(model_path):
    st.error(
        f"{selected_model} is not trained yet. "
        "Run train.py before starting the app."
    )
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# -------------------------------
# 4. Upload an MRI image
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI Image")
        st.image(image, caption="Uploaded image", width=350)

        # Resize using the same dimensions used for training.
        resized_image = image.resize((224, 224))

        # Keep pixels in the 0-255 range.
        # The selected model applies its own preprocessing.
        image_array = np.asarray(
            resized_image, dtype=np.float32
        )

        image_array = np.expand_dims(image_array, axis=0)

        if st.button("Classify Image", type="primary"):
            with st.spinner("Generating prediction..."):
                probabilities = model.predict(
                    image_array, verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = class_names[predicted_index]
                score = float(probabilities[predicted_index])

            st.subheader("Model Output")
            st.write("Predicted dataset category:")
            st.success(predicted_class)

            st.metric(
                "Model confidence score",
                f"{score * 100:.2f}%"
            )

            st.caption(
                "This score reflects the model's output, "
                "not a clinically validated probability of disease."
            )

            st.subheader("Scores for All Categories")

            for index, class_name in enumerate(class_names):
                st.write(
                    f"{class_name}: "
                    f"{probabilities[index] * 100:.2f}%"
                )

                st.progress(
                    float(probabilities[index])
                )

    except Exception as e:
        st.error(f"Could not process this image: {e}")

st.divider()

st.subheader("Project Features")
st.markdown(
    """
    - Custom Convolutional Neural Network (CNN)
    - MobileNetV2 transfer learning
    - Image resizing and normalization
    - Data augmentation
    - Accuracy, precision, recall and F1-score
    - Confusion matrix and training curves
    - Interactive image upload and prediction
    """
)
