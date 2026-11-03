"""Figura de curvas de validación y tabla comparativa de las cuatro estrategias.

Lee los historiales y resultados de 06_comparacion_arquitecturas.py (resultados/comparacion)
y los de la extracción de características (resultados/). No reentrena nada.
Salidas en resultados/comparacion: comparacion_curvas_validacion.png y tablas/comparacion_modelos.csv
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "resultados" / "comparacion"
THIAGO = BASE / "resultados"
(RES / "tablas").mkdir(exist_ok=True)


def leer(path, ep_col, offset=0):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r[ep_col]) + offset for r in rows]
    return ep, [float(r["val_accuracy"]) for r in rows], [float(r["val_loss"]) for r in rows]


series = [
    ("ANN densa", leer(RES / "historial_ann.csv", "epoch", 1), "#2a78d6", "o"),
    ("CNN desde cero", leer(RES / "historial_cnn.csv", "epoch", 1), "#eb6834", "s"),
    ("TL: extracción", leer(THIAGO / "historial_entrenamiento.csv", "epoca"), "#1baf7a", "^"),
]
n_ext = len(series[2][1][0])
series.append(("TL: fine-tuning", leer(RES / "historial_finetune.csv", "epoch", n_ext + 1), "#eda100", "D"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, idx, titulo in ((axes[0], 1, "Exactitud en validación"), (axes[1], 2, "Pérdida en validación")):
    for nombre, datos, color, marker in series:
        ax.plot(datos[0], datos[idx], color=color, lw=2, marker=marker, ms=5, label=nombre)
        i, off, ha = (-1, (4, 0), "left") if nombre != "TL: extracción" else (7, (0, 9), "center")
        ax.annotate(nombre, (datos[0][i], datos[idx][i]), xytext=off, textcoords="offset points",
                    fontsize=8, color="#333333", va="center", ha=ha)
    ax.axvline(n_ext + 0.5, color="#999999", lw=1, ls=":")
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel("Época (el fine-tuning continúa tras la época %d de la extracción)" % n_ext, fontsize=9)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0.5, n_ext + 9)
axes[0].set_ylim(0, 1)
axes[1].set_ylabel("sparse categorical cross-entropy")
axes[0].legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(RES / "comparacion_curvas_validacion.png", dpi=150)

# Tabla
filas = []
for kind, nombre in (("ann", "ANN densa"), ("cnn", "CNN desde cero")):
    d = json.loads((RES / f"resultado_{kind}.json").read_text(encoding="utf-8"))
    filas.append(d | {"nombre": nombre})
d = json.loads((RES / "resultado_finetune.json").read_text(encoding="utf-8"))
filas.append(d | {"nombre": "TL: fine-tuning"})
_, va, vl = series[2][1]
best = min(range(len(vl)), key=vl.__getitem__)
import re
met = (THIAGO / "metricas_prueba.txt").read_text(encoding="utf-8")
macro = [float(x) for x in re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", met).groups()]
resumen = (THIAGO / "resumen_modelo.txt").read_text(encoding="utf-8")
num = lambda k: int(re.search(k + r" params: ([\d,]+)", resumen).group(1).replace(",", ""))
filas.insert(2, {"nombre": "TL: extracción", "parametros_entrenables": num("Trainable"),
                 "parametros_totales": num("Total"), "mejor_epoca_val": best + 1,
                 "val_loss_mejor": vl[best], "val_accuracy_mejor": va[best],
                 "test_accuracy": float(re.search(r"Accuracy: ([\d.]+)", met).group(1)),
                 "test_precision_macro": macro[0], "test_recall_macro": macro[1], "test_f1_macro": macro[2],
                 "segundos_entrenamiento": "no registrado"})
cols = ["nombre", "parametros_entrenables", "parametros_totales", "mejor_epoca_val", "val_loss_mejor",
        "val_accuracy_mejor", "test_accuracy", "test_f1_macro", "test_precision_macro",
        "test_recall_macro", "segundos_entrenamiento"]
with open(RES / "tablas" / "comparacion_modelos.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in filas:
        w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
print((RES / "tablas" / "comparacion_modelos.csv").read_text(encoding="utf-8"))
