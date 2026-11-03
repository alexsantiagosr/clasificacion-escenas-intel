"""Exploración y preparación inicial del Intel Image Classification.

Antes de ejecutar:
  1. Instala TensorFlow y Matplotlib en el entorno de trabajo.
  2. Si el dataset está en otra ubicación, cambia DATASET_ROOT.

Este primer paso inspecciona los datos y prepara los conjuntos de entrenamiento,
validación y prueba. La carpeta seg_pred se reserva para predicciones posteriores.
"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import tensorflow as tf


# %% 1. Configuración
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("INTEL_DATASET_DIR", PROJECT_DIR.parent / "archive")).expanduser().resolve()
TRAIN_DIR = DATASET_ROOT / "seg_train" / "seg_train"
TEST_DIR = DATASET_ROOT / "seg_test" / "seg_test"
PREDICT_DIR = DATASET_ROOT / "seg_pred" / "seg_pred"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
SEED = 42


# %% 2. Comprobación de rutas
for directory in (TRAIN_DIR, TEST_DIR, PREDICT_DIR):
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe la carpeta esperada: {directory}")


# %% 3. Conteo de archivos por clase
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images_by_class(directory: Path) -> dict[str, int]:
    """Cuenta imágenes directas en cada subcarpeta de clase."""
    counts = {}
    for class_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def count_images_in_directory(directory: Path) -> int:
    """Cuenta imágenes directamente dentro de una carpeta sin etiquetas."""
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


train_counts = count_images_by_class(TRAIN_DIR)
test_counts = count_images_by_class(TEST_DIR)
prediction_count = count_images_in_directory(PREDICT_DIR)
class_names = list(train_counts)

if not class_names:
    raise ValueError(f"No se encontraron carpetas de clase en {TRAIN_DIR}")
if class_names != list(test_counts):
    raise ValueError("Las carpetas de clase de entrenamiento y prueba no coinciden.")

print("Clases:", class_names)
print("Imágenes de entrenamiento por clase:", train_counts)
print("Imágenes de prueba por clase:", test_counts)
print("Total entrenamiento:", sum(train_counts.values()))
print("Total prueba:", sum(test_counts.values()))
print("Imágenes para predicción (sin etiquetas):", prediction_count)


# %% 4. Carga de entrenamiento y validación
# La validación se separa desde seg_train. seg_test queda reservado para el
# resultado final y no participa en las decisiones tomadas durante el entrenamiento.
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    shuffle=True,
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    shuffle=True,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Orden de etiquetas:", train_ds.class_names)


# %% 5. Rendimiento del pipeline de lectura
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# %% 6. Inspección visual de un lote
images, labels = next(iter(train_ds))
plt.figure(figsize=(10, 8))
for index in range(min(9, len(images))):
    ax = plt.subplot(3, 3, index + 1)
    plt.imshow(tf.cast(images[index], tf.uint8).numpy())
    plt.title(class_names[int(labels[index])])
    plt.axis("off")
plt.tight_layout()
plt.show()


# Las imágenes están redimensionadas a 160x160 y siguen en el rango 0..255.
# En el siguiente paso se conectará este pipeline a MobileNetV2. Su preprocesamiento
# se incorporará al modelo para que la aplicación web pueda reutilizarlo sin diferencias.
