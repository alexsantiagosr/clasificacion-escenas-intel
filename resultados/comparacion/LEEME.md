# Comparación de arquitecturas

Sustenta la justificación que pide la consigna: ANN contra CNN, y entrenar desde cero contra Transfer Learning con extracción de características o con fine-tuning.

Todas las corridas usan la misma partición que `codigo/02_entrenamiento_mobilenetv2.py`:
- seg_train dividido 80/20 con semilla 42, a 160×160 y lotes de 32, que da 11.228 imágenes de entrenamiento y 2.806 de validación;
- seg_test se evalúa una sola vez por modelo y no se usa para elegir.

| Modelo | Parámetros entrenables | Mejor val_loss | Exactitud val. | Exactitud prueba | F1 macro prueba |
|---|---|---|---|---|---|
| ANN densa | 19.694.726 | 1,489 | 0,404 | 0,391 | 0,350 |
| CNN desde cero (con BatchNorm) | 110.758 | 0,483 | 0,821 | 0,821 | 0,822 |
| MobileNetV2, extracción (`modelos/modelo_escenas_mobilenetv2.keras`) | 7.686 | 0,220 | 0,919 | 0,914 | 0,915 |
| MobileNetV2, fine-tuning (`modelos/modelo_escenas_mobilenetv2_finetune.keras`) | 1.518.406 | **0,194** | **0,927** | **0,925** | **0,926** |

Las cifras salen de `tablas/comparacion_modelos.csv`, que genera `codigo/08_figura_tabla_comparacion.py`.

**Regla de elección, fijada antes de correr el experimento:** el fine-tuning reemplaza a la extracción solo si mejora el val_loss. Lo mejoró (0,194 frente a 0,220).

**Fine-tuning:** parte del modelo de extracción y descongela las últimas 30 capas de MobileNetV2, con las BatchNorm congeladas. Usa Adam 1e-5 durante 6 épocas.

**Los modelos guardados son los de la mejor época.** Se comprobó evaluándolos en validación: el de extracción da 0,21986, que corresponde a la época 14, y el de fine-tuning da 0,19402, que corresponde a la época 5.

## Archivos

- `comparacion_curvas_validacion.png`: exactitud y pérdida en validación por época de los cuatro modelos.
- `historial_*.csv` y `resultado_*.json`: el historial de cada época, más métricas, matriz de confusión y F1 por clase.
- `muestra_seg_pred_{extraccion,finetune}.png/json`: 16 imágenes de seg_pred tomadas al azar con semilla 42, con la clase y la confianza que da cada modelo.

## Limitaciones

- El fine-tuning suma 6 épocas a las 15 de la extracción, así que parte de la mejora puede venir de entrenar más.
- La ANN, con 10 épocas, y la CNN, con 15, no llegaron a converger.
- Hay una sola corrida con una sola semilla, y la diferencia en prueba es de 1,1 puntos sobre 3.000 imágenes.
- Los modelos se entrenaron en CPU. Los tiempos de `resultado_*.json` dependen del equipo.

## Reproducir

```bash
set INTEL_DATASET_DIR=<carpeta con seg_train/ seg_test/ seg_pred/>
python codigo/06_comparacion_arquitecturas.py ann cnn finetune
python codigo/07_muestra_seg_pred.py modelos/modelo_escenas_mobilenetv2_finetune.keras finetune
python codigo/08_figura_tabla_comparacion.py
```
