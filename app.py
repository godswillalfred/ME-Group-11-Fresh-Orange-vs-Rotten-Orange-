"""GET 324 Laboratory Exercise 10 application for Group ME11."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from inference_utils import display_label, interpret_probability, prepare_image


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "fresh_rotten_orange_model.keras"
CONFIG_PATH = PROJECT_DIR / "model_info.json"
DEFAULT_CONFIG = {
    "class_names": ["fresh_orange", "rotten_orange"],
    "image_size": [224, 224],
    "threshold": 0.5,
    "minimum_confidence": 0.75,
}


def load_config() -> dict:
    """Load and validate the settings produced by the Colab notebook."""
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {"class_names", "image_size", "threshold"}
    if not required.issubset(config):
        raise ValueError(
            "model_info.json must contain class_names, image_size and threshold."
        )
    if config["class_names"] != ["fresh_orange", "rotten_orange"]:
        raise ValueError(
            "The expected class order is ['fresh_orange', 'rotten_orange']."
        )
    config.setdefault("minimum_confidence", 0.75)
    return config


@st.cache_resource(show_spinner="Loading the trained model...")
def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load the Keras model once and reuse it across Streamlit reruns."""
    return tf.keras.models.load_model(model_path, compile=False)


def predict_image(
    model: tf.keras.Model,
    image: Image.Image,
    config: dict,
) -> tuple[str, float, float]:
    """Preprocess one image and return its label, confidence and raw score."""
    batch = prepare_image(image, tuple(config["image_size"]))
    rotten_probability = float(model.predict(batch, verbose=0).reshape(-1)[0])
    label, confidence = interpret_probability(
        rotten_probability,
        config["class_names"],
        float(config["threshold"]),
    )
    return label, confidence, rotten_probability


def main() -> None:
    st.set_page_config(
        page_title="Fresh or Rotten Orange",
        page_icon="🍊",
        layout="centered",
    )

    st.title("Fresh or Rotten Orange?")
    st.caption("GET 324 Laboratory Exercise 10 · Group ME11")
    st.write(
        "Upload a clear colour photograph containing one orange, then press "
        "the prediction button to estimate whether it is fresh or rotten."
    )
    st.info(
        "This binary classifier is intended only for orange images. Pictures "
        "of people, objects, other fruits or unclear scenes are outside its "
        "training task and must not be treated as reliable predictions."
    )

    with st.expander("How the application works"):
        st.write(
            "A MobileNetV3Small transfer-learning model extracts visual "
            "features from the uploaded image. Its binary output is converted "
            "into fresh-orange and rotten-orange probabilities using the "
            "decision threshold selected with validation data."
        )
        st.write(
            "A low-confidence result is rejected as uncertain. This improves "
            "safety but cannot guarantee that every unrelated image will be "
            "detected, because confidence is not proof that an orange is present."
        )

    uploaded_file = st.file_uploader(
        "Choose an orange image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG and WEBP.",
    )
    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"The uploaded file could not be opened as an image: {error}")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    if not st.button(
        "Analyse Orange",
        type="primary",
        use_container_width=True,
    ):
        st.caption("Press “Analyse Orange” when you are ready.")
        return

    if not MODEL_PATH.exists():
        st.error(
            "The trained model is missing. Run the Colab notebook, download "
            "fresh_rotten_orange_model.keras, and place it beside app.py."
        )
        return

    try:
        config = load_config()
        model = load_trained_model(str(MODEL_PATH))
        with st.spinner("Analysing the image..."):
            label, confidence, rotten_probability = predict_image(
                model, image, config
            )
    except Exception as error:
        st.exception(error)
        return

    fresh_probability = 1.0 - rotten_probability
    minimum_confidence = float(config.get("minimum_confidence", 0.75))

    if confidence < minimum_confidence:
        st.warning(
            "The result is uncertain. Upload a clearer, closer photograph of "
            "one fresh or rotten orange."
        )
        st.metric("Highest model score", f"{confidence * 100:.2f}%")
    else:
        st.success(f"Prediction: {display_label(label)}")
        st.metric("Model confidence", f"{confidence * 100:.2f}%")
        st.progress(int(round(confidence * 100)))

    st.write(
        {
            "Fresh orange probability": f"{fresh_probability * 100:.2f}%",
            "Rotten orange probability": f"{rotten_probability * 100:.2f}%",
        }
    )


if __name__ == "__main__":
    main()
