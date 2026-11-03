"""Demostración de inferencia sobre una imagen sin etiqueta de seg_pred."""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf


# %% 1. Rutas y clases
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("INTEL_DATASET_DIR", PROJECT_DIR.parent / "archive")).expanduser().resolve()
PREDICT_DIR = DATASET_ROOT / "seg_pred" / "seg_pred"
MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_escenas_mobilenetv2.keras"
RESULTS_DIR = PROJECT_DIR / "resultados"
IMAGE_SIZE = (160, 160)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
class_names = json.loads((PROJECT_DIR / "modelos" / "clases.json").read_text(encoding="utf-8"))

image_paths = sorted(
    path
    for path in PREDICT_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)
if not image_paths:
    raise FileNotFoundError(f"No se encontraron imágenes para predecir en {PREDICT_DIR}")


# %% 2. Carga de una muestra y del modelo final
# Selección determinista: se usa el primer archivo por nombre para que la
# demostración pueda repetirse y documentarse.
sample_path = image_paths[0]
model = tf.keras.models.load_model(MODEL_PATH)
image = tf.keras.utils.load_img(sample_path, target_size=IMAGE_SIZE, interpolation="bilinear")
image_array = tf.keras.utils.img_to_array(image).astype("float32")


# %% 3. Predicción y principales alternativas
probabilities = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
top_indices = np.argsort(probabilities)[::-1][:3]
predicted_index = int(top_indices[0])
predicted_class = class_names[predicted_index]
confidence = float(probabilities[predicted_index])

print("Archivo:", sample_path.name)
print("Clase predicha:", predicted_class)
print(f"Confianza del modelo: {confidence:.2%}")
print("Tres clases con mayor puntuación:")
for index in top_indices:
    print(f"  {class_names[int(index)]}: {probabilities[index]:.2%}")


# %% 4. Guardado de evidencia visual y textual
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(image)
ax.set_title(f"Predicción: {predicted_class} ({confidence:.1%})")
ax.axis("off")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "prediccion_muestra_seg_pred.png", dpi=160, bbox_inches="tight")
plt.show()

details = [
    f"Archivo: {sample_path.relative_to(DATASET_ROOT)}",
    f"Clase predicha: {predicted_class}",
    f"Puntuación/confianza reportada por el modelo: {confidence:.6f}",
    "Nota: seg_pred no contiene etiquetas; no se puede confirmar si esta predicción es correcta.",
    "Puntuaciones de las 3 clases principales:",
]
details.extend(f"{class_names[int(index)]}: {probabilities[index]:.6f}" for index in top_indices)
(RESULTS_DIR / "prediccion_muestra_seg_pred.txt").write_text("\n".join(details), encoding="utf-8")
