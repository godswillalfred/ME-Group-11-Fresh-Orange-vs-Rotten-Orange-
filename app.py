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


def render_custom_css() -> None:
    """Apply a custom orange-and-white visual theme for the app."""
    st.markdown(
        """
        <style>
            :root {
                --accent: #ff7a1a;
                --accent-dark: #cc4f00;
                --cream: #fffaf5;
                --card: rgba(255, 255, 255, 0.92);
                --text: #4c2800;
            }
            .stApp {
                background: linear-gradient(135deg, #fff7ed 0%, #fffaf5 45%, #ffe7d0 100%);
            }
            .hero-card {
                padding: 1.3rem 1.4rem;
                border-radius: 24px;
                background: linear-gradient(120deg, var(--accent) 0%, #ffae4d 100%);
                color: white;
                box-shadow: 0 10px 30px rgba(204, 79, 0, 0.2);
                margin-bottom: 1rem;
            }
            .panel {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                background: var(--card);
                border: 1px solid rgba(255, 122, 26, 0.16);
                box-shadow: 0 8px 24px rgba(255, 122, 26, 0.08);
                margin-bottom: 1rem;
            }
            .result-card {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(255, 122, 26, 0.12), rgba(255, 255, 255, 0.92));
                border: 1px solid rgba(255, 122, 26, 0.16);
                box-shadow: 0 8px 24px rgba(255, 122, 26, 0.08);
            }
            .stButton > button {
                border-radius: 999px;
                background: linear-gradient(90deg, var(--accent) 0%, #ff9e3d 100%);
                color: white;
                border: none;
                padding: 0.6rem 1.2rem;
            }
            .stButton > button:hover {
                border: none;
                background: linear-gradient(90deg, var(--accent-dark) 0%, #ff8e2b 100%);
            }
            .stFileUploader > div {
                border: 2px dashed #ffb36b;
                border-radius: 16px;
                background: rgba(255, 250, 245, 0.9);
            }
            div[data-testid="stMetric"] {
                background: white;
                border-radius: 16px;
                border: 1px solid rgba(255, 122, 26, 0.16);
                padding: 0.65rem 0.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    render_custom_css()

    st.markdown(
        """
        <div class="hero-card">
            <div style="display:flex; align-items:center; gap:0.8rem; flex-wrap:wrap;">
                <span style="font-size: 2rem;">🍊</span>
                <div>
                    <h1 style="margin:0 0 0.2rem 0; font-size: 2rem;">Fresh or Rotten Orange?</h1>
                    <p style="margin:0; opacity:0.95;">A bright, modern interface for spotting orange freshness in seconds.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro_col, tip_col = st.columns([1.55, 1.0], gap="large")

    with intro_col:
        st.markdown(
            """
            <div class="panel">
                <h3 style="margin-top:0; color:#cc4f00;">What this experience does</h3>
                <p style="margin-bottom:0; color:#5e3500;">
                    Upload a clear photo of a single orange and let the classifier estimate whether it looks fresh or rotten.
                    The experience is designed to feel calm, guided and visually clear from the first click.
                </p>
            </div>
            """,a
            unsafe_allow_html=True,
        )

    with tip_col:
        st.markdown(
            """
            <div class="panel">
                <h3 style="margin-top:0; color:#cc4f00;">Quick guidance</h3>
                <ul style="padding-left: 1rem; color:#5e3500; margin-bottom:0;">
                    <li>Use a sharp, well-lit photo.</li>
                    <li>Keep the orange as the main subject.</li>
                    <li>Low-confidence results are flagged for review.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel">
            <h3 style="margin-top:0; color:#cc4f00;">Upload an orange image</h3>
            <p style="margin-top:0; color:#5e3500;">Choose a JPG, JPEG, PNG or WEBP file to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
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

    result_icon = "✅" if label == "fresh_orange" else "⚠️"
    result_title = (
        f"{result_icon} Fresh Orange"
        if label == "fresh_orange"
        else f"{result_icon} Rotten Orange"
    )

    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="margin:0 0 0.25rem 0; color:#cc4f00;">{result_title}</h3>
            <p style="margin:0; color:#5e3500;">The model evaluated your image with a confidence score of {confidence * 100:.2f}%.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence < minimum_confidence:
        st.warning(
            "The result is uncertain. Upload a clearer, closer photograph of "
            "one fresh or rotten orange."
        )
        st.metric("Highest model score", f"{confidence * 100:.2f}%")
    else:
        st.success("Prediction is ready for review.")
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
