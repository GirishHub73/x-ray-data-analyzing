from pathlib import Path
import json
import streamlit as st
import tensorflow as tf
import yaml
from PIL import Image
from src.gradcam import (
    create_overlay,
    make_gradcam_heatmap,
    prepare_image,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
@st.cache_resource
def load_application_model(model_path):
    return tf.keras.models.load_model(model_path)
def main():
    st.set_page_config(
        page_title="Chest X-Ray Screening Prototype",
        layout="centered",
    )
    st.title("Chest X-Ray Mass/Nodule Screening Prototype")
    st.warning(
        "Educational research prototype only. This application does not diagnose cancer, provide medical advice, or support clinical decisions."
    )
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    model_path = PROJECT_ROOT / config["model"]["path"]
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        return
    model = load_application_model(model_path)
    threshold_path = PROJECT_ROOT / "results" / "selected_threshold.json"
    threshold = config["model"]["threshold"]
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as file:
            threshold = float(json.load(file)["threshold"])
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is None:
        st.info("Upload an image to run the prototype.")
        return
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded chest X-ray", use_container_width=True)
    if st.button("Analyze image"):
        image_array, display_image = prepare_image(
            image,
            config["data"]["image_size"],
        )
        probability = float(
            model.predict(image_array, verbose=0)[0][0]
        )
        flag = (
            "Needs expert review"
            if probability >= threshold
            else "Lower probability"
        )
        st.metric(
            "Possible mass/nodule probability",
            f"{probability:.1%}",
        )
        st.subheader(f"Screening flag: {flag}")
        st.caption(f"Decision threshold: {threshold:.2f}")
        try:
            heatmap = make_gradcam_heatmap(model, image_array)
            overlay = create_overlay(display_image, heatmap)
            st.image(
                overlay,
                caption="Grad-CAM visualization. Highlighted regions influenced the prediction.",
                use_container_width=True,
            )
        except Exception as error:
            st.warning(f"Grad-CAM could not be generated: {error}")
        st.info(
            "A positive flag means the model produced a higher probability based on its training data. It does not mean that cancer is present."
        )
if __name__ == "__main__":
    main()