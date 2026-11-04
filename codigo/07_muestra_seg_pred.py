"""Grilla de 16 predicciones sobre seg_pred (sin etiquetas) con clase y confianza.

Muestra aleatoria reproducible (semilla 42). Mismo preprocesamiento que la app:
RGB, redimensionado bilineal a 160x160, normalización integrada al modelo.
Uso: python 07_muestra_seg_pred.py <ruta_modelo.keras> <nombre_salida>
"""

import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Usa el mismo valor por defecto que los scripts de preparación; puede
# sustituirse definiendo INTEL_DATASET_DIR en la consola.
PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("INTEL_DATASET_DIR", PROJECT_DIR.parent / "archive")).expanduser().resolve()
PRED_DIR = ROOT / "seg_pred" / "seg_pred"
OUT = Path(os.environ.get("SALIDA", Path(__file__).resolve().parents[1] / "resultados" / "comparacion"))
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# seg_pred no tiene etiquetas reales: estas imágenes ilustran predicciones,
# pero no sirven para calcular exactitud ni confirmar si una salida acertó.
model = tf.keras.models.load_model(sys.argv[1])
name = sys.argv[2]
paths = sorted(PRED_DIR.glob("*.jpg"))
random.Random(42).shuffle(paths)
sample = paths[:16]

# La aplicación hace el mismo preprocesamiento RGB y 160x160. La normalización
# a [-1, 1] ocurre dentro del modelo guardado, así que no se repite aquí.
images = np.stack([
    np.asarray(tf.keras.utils.load_img(p, target_size=(160, 160), interpolation="bilinear"), dtype=np.float32)
    for p in sample
])
probs = model.predict(images, verbose=0)

rows = []
fig, axes = plt.subplots(4, 4, figsize=(12, 13))
for ax, path, img, pr in zip(axes.flat, sample, images, probs):
    order = pr.argsort()[::-1]
    top, second = order[0], order[1]
    rows.append({"archivo": path.name, "clase": CLASSES[top], "confianza": round(float(pr[top]), 4),
                 "segunda": CLASSES[second], "conf_segunda": round(float(pr[second]), 4)})
    ax.imshow(img.astype(np.uint8))
    ax.set_title(f"{path.name}\n{CLASSES[top]} {pr[top]:.0%}  (2.ª {CLASSES[second]} {pr[second]:.0%})", fontsize=9)
    ax.axis("off")
fig.suptitle("Predicciones sobre seg_pred (sin etiqueta real)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / f"muestra_seg_pred_{name}.png", dpi=120)
(OUT / f"muestra_seg_pred_{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=1))
