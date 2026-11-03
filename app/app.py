"""Interfaz web para clasificar escenas con el modelo MobileNetV2 entrenado."""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_escenas_mobilenetv2.keras"
CLASSES_PATH = PROJECT_DIR / "modelos" / "clases.json"
IMAGE_SIZE = (160, 160)

st.set_page_config(page_title="Clasificador de escenas", page_icon="🌄", layout="centered")


@st.cache_resource
def load_model_and_classes():
    """Carga una sola vez el modelo y verifica el orden de sus clases."""
    import tensorflow as tf

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encuentra el modelo entrenado: {MODEL_PATH}")
    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(f"No se encuentra el archivo de clases: {CLASSES_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            f"Hay {len(class_names)} nombres de clase, pero el modelo produce "
            f"{model.output_shape[-1]} salidas."
        )
    return model, class_names


st.title("Clasificador de escenas")
st.write(
    "Carga una fotografía y el modelo estimará si corresponde a edificios, bosque, "
    "glaciar, montaña, mar o calle."
)
st.caption(
    "La imagen se convierte a RGB y se redimensiona a 160 × 160. "
    "La normalización de MobileNetV2 ya está integrada en el modelo."
)

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        image_array = np.asarray(resized_image, dtype=np.float32)

        with st.spinner("Cargando el modelo y clasificando la imagen..."):
            model, class_names = load_model_and_classes()
            probabilities = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]

        ranked_indices = np.argsort(probabilities)[::-1]
        predicted_index = int(ranked_indices[0])

        image_column, result_column = st.columns([1.15, 1])
        with image_column:
            st.image(original_image, caption="Imagen recibida", width="stretch")
        with result_column:
            st.subheader("Resultado")
            st.metric("Clase predicha", class_names[predicted_index])
            st.metric("Puntuación softmax", f"{probabilities[predicted_index]:.1%}")

        st.subheader("Puntuaciones por clase")
        for index in ranked_indices:
            score = float(probabilities[index])
            label_column, bar_column = st.columns([1.2, 4])
            label_column.write(class_names[int(index)])
            bar_column.progress(score, text=f"{score:.1%}")

        st.caption(
            "La puntuación softmax es la salida del modelo y no garantiza que la confianza "
            "esté calibrada como una probabilidad real."
        )
    except Exception as error:
        st.error(f"No se pudo procesar la imagen: {error}")
