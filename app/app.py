"""Interfaz en español para clasificar escenas con MobileNetV2."""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_escenas_mobilenetv2.keras"
CLASSES_PATH = PROJECT_DIR / "modelos" / "clases.json"
IMAGE_SIZE = (160, 160)

CLASS_LABELS = {
    "buildings": "Edificios",
    "forest": "Bosque",
    "glacier": "Glaciar",
    "mountain": "Montaña",
    "sea": "Mar",
    "street": "Calle",
}

CLASS_EXPLANATIONS = {
    "buildings": "Escena urbana donde edificios u otras construcciones forman parte destacada del paisaje.",
    "forest": "Paisaje natural con árboles y vegetación como elementos predominantes.",
    "glacier": "Paisaje de hielo o nieve asociado a un entorno glaciar.",
    "mountain": "Paisaje de montaña o relieve elevado.",
    "sea": "Escena de mar o costa donde el agua es el elemento principal.",
    "street": "Escena de una calle o vía urbana y su entorno construido.",
}

st.set_page_config(
    page_title="Clasificador de escenas",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Usa el tema elegido por quien ejecuta la aplicación. */

      .hero-card {
        padding: 1.5rem 1.7rem; border-radius: 20px;
        background: linear-gradient(120deg, #123b4a, #176b67);
        color: #ffffff; margin: .4rem 0 1.3rem 0;
        box-shadow: 0 12px 32px rgba(18,59,74,.14);
      }
      .hero-card p { color: #e4f2f1; margin-bottom: 0; font-size: 1.04rem; }
      .result-card {
        padding: 1.1rem 1.25rem; border-radius: 16px;
        background: #eef7f5; border: 1px solid #cce7e0;
        margin: .5rem 0 1rem 0;
      }
      .result-label { color: #35615b; font-size: .9rem; margin-bottom: .2rem; }
      .result-value { color: #123b4a; font-size: 1.8rem; font-weight: 750; }
      .small-note { color: #52636a; font-size: .9rem; }
      [data-testid="stMetric"] { padding: .8rem 1rem; border-radius: 14px; border: 1px solid rgba(128,150,160,.28); }
      [data-testid="stFileUploaderDropzone"] button p { font-size: 0; }
      [data-testid="stFileUploaderDropzone"] button p::after { content: "Elegir archivo"; font-size: .9rem; }
      [data-testid="stFileUploaderDropzone"] small { font-size: 0; }
      [data-testid="stFileUploaderDropzone"] small::after { content: "Hasta 200 MB · JPG o PNG"; font-size: .75rem; }
      @media (max-width: 700px) {
        .hero-card { padding: 1.1rem; }
        .result-value { font-size: 1.45rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_classes():
    """Carga el modelo una vez y verifica el orden de las seis clases."""
    import tensorflow as tf

    if not MODEL_PATH.is_file():
        raise FileNotFoundError("No se encontró el modelo entrenado en la carpeta modelos.")
    if not CLASSES_PATH.is_file():
        raise FileNotFoundError("No se encontró el archivo de nombres de clase.")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    if len(class_names) != model.output_shape[-1]:
        raise ValueError("El modelo y el archivo de nombres no tienen la misma cantidad de clases.")
    return model, class_names


st.markdown(
    """
    <div class="hero-card">
      <h1>Clasificador de escenas</h1>
      <p>Sube una fotografía o tómala con la cámara para estimar cuál de seis categorías describe mejor la escena.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(
    "El sistema analiza la imagen completa. Clasifica el tipo de escena; no detecta ni marca objetos individuales."
)

with st.sidebar:
    st.header("Sobre este modelo")
    st.write("**Arquitectura:** CNN MobileNetV2 con transferencia de aprendizaje.")
    st.write("**Estrategia:** extracción de características; la base preentrenada se mantuvo congelada.")
    st.write("**Categorías:** edificios, bosque, glaciar, montaña, mar y calle.")
    st.metric("Exactitud en prueba (accuracy)", "91,37 %", help="Evaluado una vez sobre 3.000 imágenes reservadas.")
    st.metric("F1 macro en prueba", "91,54 %", help="Promedio que da el mismo peso a cada una de las seis clases.")
    st.caption("La evaluación final se hizo con seg_test, separado del entrenamiento y la validación.")

with st.expander("¿Cómo prepara y procesa la imagen?", expanded=False):
    st.markdown(
        """
        1. Convierte la fotografía cargada a formato RGB.
        2. La redimensiona a **160 × 160 píxeles**, igual que el tamaño de entrada usado al entrenar.
        3. El propio modelo aplica la normalización de píxeles requerida por MobileNetV2.
        4. La red compara el patrón global con las seis clases aprendidas y ordena sus puntuaciones.

        Durante el entrenamiento se separó `seg_train` en entrenamiento y validación (80/20),
        se aplicó aumento de datos solo al entrenamiento y se reservó `seg_test` para la evaluación final.
        """
    )

input_method = st.radio(
    "¿Cómo quieres añadir la fotografía?",
    ["Subir desde el dispositivo", "Tomar foto con la cámara"],
    horizontal=True,
    help="En un celular, la opción de cámara puede abrir la cámara o el selector del navegador.",
)

if input_method == "Tomar foto con la cámara":
    uploaded_file = st.camera_input(
        "Toma una fotografía de la escena",
        key="camera_input",
        help="Al capturarla, la imagen se carga en la aplicación para clasificarla.",
    )
    image_caption = "Fotografía tomada con la cámara"
else:
    uploaded_file = st.file_uploader(
        "Selecciona una fotografía",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Formatos admitidos: JPG, JPEG y PNG.",
    )
    image_caption = "Fotografía subida desde el dispositivo"

if uploaded_file is None:
    st.info("Sube una imagen o toma una foto para ver la categoría estimada, las alternativas y una explicación del resultado.")
else:
    try:
        original_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
        resized_image = original_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        image_array = np.asarray(resized_image, dtype=np.float32)

        with st.spinner("Cargando el modelo y analizando la escena…"):
            model, class_names = load_model_and_classes()
            scores = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]

        ranked_indices = np.argsort(scores)[::-1]
        first_index = int(ranked_indices[0])
        second_index = int(ranked_indices[1])
        predicted_class = class_names[first_index]
        predicted_label = CLASS_LABELS.get(predicted_class, predicted_class.title())
        top_score = float(scores[first_index])
        second_score = float(scores[second_index])
        score_gap = top_score - second_score

        image_column, prediction_column = st.columns([1.05, 1.15], gap="large")
        with image_column:
            st.image(original_image, caption=image_caption, width="stretch")
        with prediction_column:
            st.subheader("Resultado de la clasificación")
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-label">Categoría con mayor puntuación</div>
                  <div class="result-value">{predicted_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Puntuación softmax", f"{top_score:.1%}")
            st.write(CLASS_EXPLANATIONS.get(predicted_class, "El modelo eligió esta categoría como su salida principal."))
            st.caption("Esta descripción explica la etiqueta; no identifica los píxeles que influyeron en la decisión.")

        st.subheader("Otras categorías consideradas")
        st.caption("Las barras muestran la salida del modelo para cada categoría, ordenada de mayor a menor.")
        for index in ranked_indices:
            class_name = class_names[int(index)]
            label = CLASS_LABELS.get(class_name, class_name.title())
            score = float(scores[index])
            label_column, bar_column = st.columns([1.15, 4])
            label_column.write(label)
            bar_column.progress(score, text=f"{score:.1%}")

        second_label = CLASS_LABELS.get(class_names[second_index], class_names[second_index].title())
        if score_gap <= 0.15:
            st.warning(
                f"La segunda alternativa, **{second_label}**, tiene una puntuación cercana. "
                "El modelo muestra cierta ambigüedad entre ambas categorías; conviene revisar la imagen."
            )
        else:
            st.info(f"La siguiente alternativa es **{second_label}** con una puntuación de {second_score:.1%}.")

        with st.expander("Cómo interpretar la puntuación y sus límites", expanded=False):
            st.write(
                "La puntuación softmax expresa cómo reparte el modelo su salida entre las seis categorías. "
                "No está calibrada como una probabilidad real de acierto y no garantiza que la etiqueta sea correcta. "
                "El modelo clasifica escenas completas; puede confundir categorías visualmente parecidas, "
                "por ejemplo montaña y glaciar, o edificios y calle."
            )
    except Exception:
        st.error("No se pudo procesar esa imagen. Comprueba que sea un archivo JPG, JPEG o PNG válido e inténtalo de nuevo.")
        st.caption("Si el problema continúa, revisa los registros de la aplicación.")
