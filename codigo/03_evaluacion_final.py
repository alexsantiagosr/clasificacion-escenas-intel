"""Evaluación final en seg_test del modelo seleccionado por validación."""

from pathlib import Path
import os
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss


# %% 1. Configuración
# Se recupera el mismo orden de clases guardado durante el entrenamiento. Es
# esencial: la salida 0, por ejemplo, solo significa la primera clase de ese orden.
# El archivo del modelo y el dataset de prueba deben existir antes de continuar.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("INTEL_DATASET_DIR", PROJECT_DIR.parent / "archive")).expanduser().resolve()
TEST_DIR = DATASET_ROOT / "seg_test" / "seg_test"
MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_escenas_mobilenetv2.keras"
RESULTS_DIR = PROJECT_DIR / "resultados"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASSES_PATH = PROJECT_DIR / "modelos" / "clases.json"
class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
test_class_names = sorted(path.name for path in TEST_DIR.iterdir() if path.is_dir())
if class_names != test_class_names:
    raise ValueError(f"El orden de clases del modelo no coincide con seg_test: {class_names} vs {test_class_names}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")


# %% 2. Carga del modelo y del conjunto de prueba
# seg_test no intervino en el ajuste de pesos ni en la selección de la época.
# shuffle=False mantiene alineadas las etiquetas con la lista de archivos para
# generar después un CSV donde cada predicción pueda rastrearse a su imagen.
# Esta evaluación usa seg_test una vez que la selección del modelo ya se hizo
# mirando únicamente val_loss y val_accuracy.
model = tf.keras.models.load_model(MODEL_PATH)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_paths = [
    path
    for class_name in class_names
    for path in sorted((TEST_DIR / class_name).iterdir())
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
]


# %% 3. Predicción de prueba
# El modelo entrega seis puntuaciones por imagen. Se concatenan los lotes para
# calcular las métricas sobre todo el conjunto. argmax elige el índice de mayor
# puntuación como clase predicha; y_true son las clases reales de las carpetas.
y_true_batches = []
y_probability_batches = []

for images, labels in test_ds:
    probabilities = model(images, training=False).numpy()
    y_true_batches.append(labels.numpy())
    y_probability_batches.append(probabilities)

y_true = np.concatenate(y_true_batches)
y_probabilities = np.concatenate(y_probability_batches)
y_predicted = np.argmax(y_probabilities, axis=1)

if len(test_paths) != len(y_true):
    raise ValueError(
        f"Cantidad de rutas ({len(test_paths)}) y predicciones ({len(y_true)}) no coincide."
    )


# %% 4. Métricas multiclase
# Accuracy es la fracción total de aciertos. Log loss penaliza más las predicciones
# muy seguras pero equivocadas. classification_report calcula precision (aciertos
# entre predicciones de una clase), recall (aciertos encontrados entre los casos
# reales de esa clase) y F1 (equilibrio entre ambas), además de promedios. La matriz
# de confusión cuenta las clases reales frente a las predichas.
label_ids = list(range(len(class_names)))
accuracy = accuracy_score(y_true, y_predicted)
test_crossentropy = log_loss(y_true, y_probabilities, labels=label_ids)
report = classification_report(
    y_true,
    y_predicted,
    labels=label_ids,
    target_names=class_names,
    digits=4,
    zero_division=0,
)
matrix = confusion_matrix(y_true, y_predicted, labels=label_ids)

summary = (
    f"Número de imágenes de prueba: {len(y_true)}\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"Cross-entropy (log loss): {test_crossentropy:.4f}\n\n"
    "Métricas por clase y promedios:\n"
    f"{report}"
)
print(summary)
(RESULTS_DIR / "metricas_prueba.txt").write_text(summary, encoding="utf-8")


# %% 5. Matriz de confusión
# Las filas representan la clase real y las columnas la predicha. Los valores de
# la diagonal son aciertos; fuera de ella quedan los tipos de confusión. Se guarda
# la figura para incluirla como evidencia en el análisis y en el informe.
fig, ax = plt.subplots(figsize=(8, 7))
image = ax.imshow(matrix, cmap="Blues")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
ax.set(
    xticks=label_ids,
    yticks=label_ids,
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Clase predicha",
    ylabel="Clase real",
    title="Matriz de confusión — conjunto de prueba",
)
plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
threshold = matrix.max() / 2 if matrix.size else 0
for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
        ax.text(
            column,
            row,
            str(matrix[row, column]),
            ha="center",
            va="center",
            color="white" if matrix[row, column] > threshold else "black",
        )
fig.tight_layout()
fig.savefig(RESULTS_DIR / "matriz_confusion_prueba.png", dpi=160, bbox_inches="tight")
plt.show()


# %% 6. Tabla de predicciones para revisar aciertos y errores
# Este CSV conserva para cada archivo la etiqueta verdadera, la predicción y su
# puntuación más alta. Permite investigar ejemplos concretos; la puntuación del
# modelo no garantiza por sí sola que la predicción sea correcta.
with (RESULTS_DIR / "predicciones_prueba.csv").open(
    "w", newline="", encoding="utf-8"
) as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["archivo", "clase_real", "clase_predicha", "confianza", "acierto"])
    for path, true_id, predicted_id, probabilities in zip(
        test_paths, y_true, y_predicted, y_probabilities
    ):
        writer.writerow(
            [
                str(path.relative_to(DATASET_ROOT)),
                class_names[int(true_id)],
                class_names[int(predicted_id)],
                f"{float(probabilities[predicted_id]):.6f}",
                bool(true_id == predicted_id),
            ]
        )

print("Métricas guardadas en:", RESULTS_DIR / "metricas_prueba.txt")
print("Matriz guardada en:", RESULTS_DIR / "matriz_confusion_prueba.png")
print("Predicciones guardadas en:", RESULTS_DIR / "predicciones_prueba.csv")
