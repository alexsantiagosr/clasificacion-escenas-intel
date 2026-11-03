"""Verifica que el modelo exportado y el orden de clases sean compatibles."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_escenas_mobilenetv2.keras"
CLASSES_PATH = PROJECT_DIR / "modelos" / "clases.json"

model = tf.keras.models.load_model(MODEL_PATH)
class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

if model.input_shape[1:] != (160, 160, 3):
    raise ValueError(f"Forma de entrada inesperada: {model.input_shape}")
if model.output_shape[-1] != len(class_names):
    raise ValueError(
        f"El modelo tiene {model.output_shape[-1]} salidas, "
        f"pero clases.json contiene {len(class_names)} nombres."
    )

sample = np.zeros((1, 160, 160, 3), dtype=np.float32)
output = model.predict(sample, verbose=0)
if output.shape != (1, len(class_names)) or not np.isfinite(output).all():
    raise ValueError(f"Salida no válida: forma={output.shape}, valores={output}")

print("Modelo cargado:", MODEL_PATH.name)
print("Entrada esperada:", model.input_shape)
print("Orden de clases:", class_names)
print("Salida para una imagen de comprobación:", output[0])
