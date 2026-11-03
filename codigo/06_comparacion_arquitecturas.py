"""Comparación ANN vs CNN desde cero vs Transfer Learning (extracción y fine-tuning).

Complementa el repo de Thiago (clasificacion-escenas-intel) sin tocar sus archivos.
Usa EXACTAMENTE su partición: seg_train 80/20 con semilla 42, 160x160, lotes de 32.
La selección de cada modelo se hace con validación (EarlyStopping restaura el mejor
val_loss); seg_test se evalúa una sola vez al final de cada modelo, solo para reportar.

Uso:
  python 06_comparacion_arquitecturas.py ann
  python 06_comparacion_arquitecturas.py cnn
  python 06_comparacion_arquitecturas.py finetune   (requiere el .keras de Thiago)

Variables: INTEL_DATASET_DIR (raíz con seg_train/seg_test), MODELO_BASE (ruta del .keras
de extracción de características; por defecto modelos/modelo_escenas_mobilenetv2.keras),
SALIDA (carpeta de resultados; por defecto resultados/comparacion).
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

SEED = 42
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20

ROOT = Path(os.environ["INTEL_DATASET_DIR"])
TRAIN_DIR = ROOT / "seg_train" / "seg_train"
TEST_DIR = ROOT / "seg_test" / "seg_test"
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SALIDA", PROJECT_DIR / "resultados" / "comparacion"))
OUT.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)
CLASSES = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())


def load(directory, subset=None, shuffle=True):
    kwargs = dict(labels="inferred", label_mode="int", class_names=CLASSES,
                  image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED, shuffle=shuffle)
    if subset:
        kwargs.update(validation_split=VALIDATION_SPLIT, subset=subset)
    return tf.keras.utils.image_dataset_from_directory(directory, **kwargs).prefetch(tf.data.AUTOTUNE)


train_ds = load(TRAIN_DIR, "training")
val_ds = load(TRAIN_DIR, "validation")
test_ds = load(TEST_DIR, shuffle=False)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
], name="aumento_datos")


def build_ann():
    """ANN densa: la imagen se aplana y se pierde la estructura espacial."""
    inputs = tf.keras.Input((*IMAGE_SIZE, 3))
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="ann_densa")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, 10


def build_cnn():
    """CNN desde cero, misma idea que la del profesor más BatchNorm y Dropout."""
    inputs = tf.keras.Input((*IMAGE_SIZE, 3))
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="cnn_desde_cero")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, 15


def build_finetune():
    """Parte del modelo de extracción de características y descongela el final de MobileNetV2."""
    model = tf.keras.models.load_model(os.environ.get("MODELO_BASE", PROJECT_DIR / "modelos" / "modelo_escenas_mobilenetv2.keras"))
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model) and "mobilenet" in l.name.lower())
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:  # BatchNorm congelada: evita romper las estadísticas de ImageNet
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, 6


def main(kind):
    model, epochs = {"ann": build_ann, "cnn": build_cnn, "finetune": build_finetune}[kind]()
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(model.count_params())
    print(f"[{kind}] parámetros entrenables={trainable:,} total={total:,}", flush=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(OUT / f"historial_{kind}.csv"),
    ]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    train_s = time.time() - t0

    best = int(np.argmin(hist.history["val_loss"]))
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(1)

    from sklearn.metrics import classification_report, confusion_matrix
    rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, digits=4)
    res = {
        "modelo": kind, "parametros_entrenables": trainable, "parametros_totales": total,
        "epocas_corridas": len(hist.history["loss"]), "mejor_epoca_val": best + 1,
        "val_loss_mejor": float(hist.history["val_loss"][best]),
        "val_accuracy_mejor": float(hist.history["val_accuracy"][best]),
        "test_accuracy": float(rep["accuracy"]), "test_f1_macro": float(rep["macro avg"]["f1-score"]),
        "test_precision_macro": float(rep["macro avg"]["precision"]),
        "test_recall_macro": float(rep["macro avg"]["recall"]),
        "segundos_entrenamiento": round(train_s, 1),
        "matriz_confusion": confusion_matrix(y_true, y_pred).tolist(), "clases": CLASSES,
        "por_clase_f1": {c: float(rep[c]["f1-score"]) for c in CLASSES},
    }
    (OUT / f"resultado_{kind}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    if kind == "finetune":
        # Se guarda sin el estado del optimizador (9,7 MB en vez de 21,8 MB): la app solo predice.
        model.save(OUT / "modelo_completo_finetune.keras")
        ligero = tf.keras.models.load_model(OUT / "modelo_completo_finetune.keras", compile=False)
        ligero.save(PROJECT_DIR / "modelos" / "modelo_escenas_mobilenetv2_finetune.keras")
        (OUT / "modelo_completo_finetune.keras").unlink()
    print(json.dumps({k: v for k, v in res.items() if k != "matriz_confusion"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        main(k)
