"""Primera fase: MobileNetV2 como extractor de características.

Entrena con seg_train y valida con una partición fija de seg_train. No carga ni
evalúa seg_test; esa carpeta se reserva para evaluar una vez elegido el modelo.
"""

from pathlib import Path
import os
import csv
import json
import time

import matplotlib.pyplot as plt
import tensorflow as tf


# %% 1. Configuración y rutas
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("INTEL_DATASET_DIR", PROJECT_DIR.parent / "archive")).expanduser().resolve()
TRAIN_DIR = DATASET_ROOT / "seg_train" / "seg_train"

MODEL_DIR = PROJECT_DIR / "modelos"
RESULTS_DIR = PROJECT_DIR / "resultados"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
SEED = 42
EPOCHS = 15

MODEL_PATH = MODEL_DIR / "modelo_escenas_mobilenetv2.keras"
CLASSES_PATH = MODEL_DIR / "clases.json"
HISTORY_PATH = RESULTS_DIR / "historial_entrenamiento.csv"
CURVES_PATH = RESULTS_DIR / "curvas_entrenamiento.png"
SUMMARY_PATH = RESULTS_DIR / "resumen_modelo.txt"

if not TRAIN_DIR.is_dir():
    raise FileNotFoundError(f"No existe la carpeta de entrenamiento: {TRAIN_DIR}")

class_names = sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())
if len(class_names) != 6:
    raise ValueError(f"Se esperaban 6 clases y se encontraron {len(class_names)}: {class_names}")

CLASSES_PATH.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
print("Clases en orden:", class_names)
print("Resolución de entrada:", IMAGE_SIZE)
print("Épocas máximas:", EPOCHS)


# %% 2. Carga de entrenamiento y validación
# La misma semilla y el mismo porcentaje garantizan una partición reproducible.
# Las imágenes llegan con valores 0..255; la normalización para MobileNetV2 se
# incorpora dentro del propio modelo más abajo.
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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)


# %% 3. Aumento de datos y base preentrenada
# El aumento solo se activa durante entrenamiento; las imágenes de validación
# conservan su contenido original.
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="aumento_datos",
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False


# %% 4. Clasificador para las seis escenas
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="imagen_rgb_0_255")
x = data_augmentation(inputs)
# MobileNetV2 espera los píxeles escalados entre -1 y 1.
x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="normalizacion_mobilenetv2")(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="promedio_global")(x)
x = tf.keras.layers.Dropout(0.30, name="dropout")(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="clasificador_6_clases")(x)

model = tf.keras.Model(inputs, outputs, name="escenas_mobilenetv2_transfer_learning")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

summary_lines = []
model.summary(print_fn=summary_lines.append)
summary_text = "\n".join(summary_lines)
print(summary_text)
SUMMARY_PATH.write_text(
    "Orden de clases: " + ", ".join(class_names) + "\n\n" + summary_text,
    encoding="utf-8",
)


# %% 5. Entrenamiento con control de sobreajuste
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

training_started = time.perf_counter()
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    shuffle=False,  # Los datos ya se mezclaron al construir train_ds.
)

# EarlyStopping restaura los mejores pesos de validación antes de guardar.
training_elapsed_seconds = time.perf_counter() - training_started
elapsed_hours, elapsed_remainder = divmod(training_elapsed_seconds, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
training_elapsed_text = f"{int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{elapsed_seconds:05.2f}"
model.save(MODEL_PATH)


# %% 6. Guardado de historial y curvas
history_dict = history.history
best_epoch = min(range(len(history_dict["val_loss"])), key=history_dict["val_loss"].__getitem__)

with HISTORY_PATH.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["epoca", "loss", "accuracy", "val_loss", "val_accuracy"])
    for index, (loss, accuracy, val_loss, val_accuracy) in enumerate(
        zip(
            history_dict["loss"],
            history_dict["accuracy"],
            history_dict["val_loss"],
            history_dict["val_accuracy"],
        ),
        start=1,
    ):
        writer.writerow([index, loss, accuracy, val_loss, val_accuracy])

epochs_ran = range(1, len(history_dict["loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(epochs_ran, history_dict["accuracy"], label="Entrenamiento")
axes[0].plot(epochs_ran, history_dict["val_accuracy"], label="Validación")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Accuracy")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(epochs_ran, history_dict["loss"], label="Entrenamiento")
axes[1].plot(epochs_ran, history_dict["val_loss"], label="Validación")
axes[1].set_title("Pérdida")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.tight_layout()
fig.savefig(CURVES_PATH, dpi=150, bbox_inches="tight")
plt.show()

print(f"Mejor época según val_loss: {best_epoch + 1}")
print(f"val_loss en la mejor época: {history_dict['val_loss'][best_epoch]:.4f}")
print(f"val_accuracy en la mejor época: {history_dict['val_accuracy'][best_epoch]:.4f}")
print(f"Modelo guardado en: {MODEL_PATH}")
print(f"Historial guardado en: {HISTORY_PATH}")
print(f"Curvas guardadas en: {CURVES_PATH}")
SUMMARY_PATH.write_text(
    "Orden de clases: " + ", ".join(class_names) + "\n\n" + summary_text
    + f"\n\nDuración de model.fit: {training_elapsed_text} ({training_elapsed_seconds:.2f} segundos).",
    encoding="utf-8",
)
print(f"Duración de model.fit: {training_elapsed_text} ({training_elapsed_seconds:.2f} segundos)")
