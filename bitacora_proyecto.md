# Bitácora del proyecto

**Proyecto:** Clasificación de escenas naturales y urbanas con Intel Image Classification
**Última actualización:** 3 de noviembre de 2026

Esta bitácora registra decisiones, procedimientos, resultados y cambios para usar como fuente del informe CRISP-DM y de la sustentación. Se actualizará con resultados reales a medida que se complete cada etapa.

## 1. Comprensión del negocio

La empresa de turismo y gestión territorial necesita organizar automáticamente fotografías de escenas. El modelo debe asignar a cada imagen una de seis categorías: edificios, bosque, glaciar, montaña, mar o calle. El proyecto también requiere una aplicación web que reciba una fotografía y muestre la clase predicha junto con su confianza.

## 2. Comprensión de los datos

- Fuente: Intel Image Classification, descargado desde el enlace indicado por el profesor.
- Imágenes observadas: archivos JPEG a color RGB de 150 × 150 píxeles.
- Estructura descargada: `archive/seg_train/seg_train`, `archive/seg_test/seg_test` y `archive/seg_pred/seg_pred`.
- Entrenamiento: 14.034 imágenes en seis carpetas de clase.
- Prueba: 3.000 imágenes en seis carpetas de clase.
- Predicción: 7.301 imágenes sin etiqueta, directamente en la carpeta de predicción.
- Conteo de entrenamiento por clase: buildings 2.191; forest 2.271; glacier 2.404; mountain 2.512; sea 2.274; street 2.382.
- Conteo de prueba por clase: buildings 437; forest 474; glacier 553; mountain 525; sea 510; street 501.

## 3. Preparación de los datos

- Se conservará `seg_test` para la evaluación final; no se usará como validación durante el entrenamiento.
- Se separó el 20% de `seg_train` para validación mediante `image_dataset_from_directory`, usando semilla 42.
- Resultado de la partición: 11.228 imágenes de entrenamiento y 2.806 de validación.
- El pipeline redimensiona las imágenes a 160 × 160 y trabaja con lotes de 32 imágenes.
- El dataset `seg_pred` se reserva para ejemplos de predicción, no para calcular métricas al no tener etiquetas.
- Se aplicó aumento de datos solo al entrenamiento. La normalización MobileNetV2 se incorporó al modelo y la inferencia conserva el mismo redimensionamiento RGB de 160 × 160.

## 4. Modelado: decisiones iniciales

- Arquitectura elegida: CNN, porque las imágenes tienen patrones espaciales que las convoluciones pueden aprender.
- Estrategia propuesta: Transfer Learning con MobileNetV2 preentrenada en ImageNet.
- Primera fase prevista: congelar la base y entrenar una nueva cabeza clasificadora para seis clases.
- Se considerará ajuste fino de algunas capas según los resultados de validación.
- Estas son decisiones de diseño iniciales; se documentarán los cambios y motivos si los resultados llevan a modificarlas.

## 5. Entorno y verificación inicial

- Desarrollo local en VS Code/Jupyter sobre Windows.
- Entorno virtual del proyecto: `.venv`.
- Python observado: 3.13.5.
- TensorFlow importado correctamente: versión 2.21.0.
- El script `codigo/01_preparacion_datos.py` se ejecutó y encontró las seis clases, las cantidades indicadas y los tres conjuntos.
- Los mensajes de oneDNN observados al importar TensorFlow son informativos; la ejecución no mostró error de carga.

## 6. Evaluación

Completado: las métricas de prueba, la matriz de confusión, las curvas de entrenamiento/validación y los análisis de errores están guardados en `resultados/`. Falta medir retrospectivamente la duración real del entrenamiento, que no se registró en la primera ejecución.

## 7. Despliegue

Completado en el entorno local: modelo `.keras`, preprocesamiento coherente, app Streamlit y predicción de muestra guardados. Despliegue de la versión anterior publicado; la interfaz local actualizada no se ha publicado y la inferencia de una imagen en la nube sigue pendiente de prueba.

## Registro de experimentos

Se añadirá una entrada por cada entrenamiento con fecha, resolución, aumentos, arquitectura, capas congeladas/descongeladas, optimizador, tasa de aprendizaje, épocas, callbacks y métricas de validación/prueba.

## Configuración del primer experimento de modelado

Se preparó y ejecutó `codigo/02_entrenamiento_mobilenetv2.py`. El experimento usa MobileNetV2 preentrenada en ImageNet con la base congelada, aumento aleatorio horizontal/rotacional/zoom en entrenamiento, normalización integrada al modelo al rango -1..1, una cabeza de pooling global + Dropout(0.30) + softmax de seis clases, Adam con tasa 0.001, sparse categorical crossentropy y máximo 15 épocas. Incluye ModelCheckpoint, EarlyStopping (paciencia 4) y ReduceLROnPlateau (paciencia 2).

Usará solo entrenamiento y validación. No leerá `seg_test`; ese conjunto se mantiene reservado para la evaluación final. Guardará el modelo `.keras`, resumen de arquitectura, historial CSV y curvas accuracy/loss en las carpetas `modelos/` y `resultados/`.

## Resultados del experimento 1: extracción de características

- Fecha de ejecución: 2 de noviembre de 2026.
- Se usaron 11.228 imágenes para entrenar y 2.806 para validar, redimensionadas a 160 × 160, en lotes de 32.
- MobileNetV2 preentrenada en ImageNet, con la base congelada; solo 7.686 parámetros entrenables de un total de 2.265.670.
- Aumento solo durante entrenamiento: volteo horizontal, rotación 0,05 y zoom 0,10.
- Preprocesamiento integrado al modelo: escalado de píxeles de 0..255 a -1..1.
- Adam (0,001 inicial), sparse categorical crossentropy, Dropout 0,30, ModelCheckpoint, EarlyStopping (paciencia 4) y ReduceLROnPlateau (paciencia 2).
- El entrenamiento llegó al límite de 15 épocas. El mejor checkpoint por `val_loss` fue la época 14: `val_loss` 0,2199 y `val_accuracy` 0,9191 (91,91%).
- En la época 15: accuracy de entrenamiento 0,9098; loss 0,2410; `val_accuracy` 0,9198; `val_loss` 0,2200.
- No se usó `seg_test` para seleccionar el modelo.
- Archivos generados: `modelos/modelo_escenas_mobilenetv2.keras`, `resultados/historial_entrenamiento.csv`, `resultados/curvas_entrenamiento.png` y `resultados/resumen_modelo.txt`.
- Lectura preliminar: la validación se mantuvo estable alrededor de 92%; no se observa separación grande que sugiera sobreajuste. El accuracy de entrenamiento se mide con aumento aleatorio y dropout activos, por eso puede quedar por debajo de validación.

## Evaluación final

Se ejecutó `codigo/03_evaluacion_final.py` una vez sobre las 3.000 imágenes de `seg_test`; el modelo seleccionado fue el checkpoint de validación. Las métricas, matriz de confusión y predicciones quedaron guardadas en resultados.

## Resultados de evaluación final en seg_test

- Evaluación realizada una vez sobre las 3.000 imágenes de prueba después de seleccionar el checkpoint por validación.
- Accuracy: 0,9137 (91,37%).
- Log loss: 0,2375.
- Promedio macro: precision 0,9155; recall 0,9157; F1 0,9154.
- Promedio ponderado: precision 0,9134; recall 0,9137; F1 0,9133.
- F1 por clase: buildings 0,9171; forest 0,9884; glacier 0,8553; mountain 0,8585; sea 0,9450; street 0,9279.
- Mejores resultados por clase: forest (recall 0,9916) y sea (recall 0,9608).
- Las clases con más confusión son glacier y mountain: 48 glaciares se predijeron como montaña y 71 montañas como glaciar. También hubo 35 edificios predichos como calle y 29 calles como edificios.
- Ejemplo concreto para discutir el error: `seg_test/seg_test/mountain/20071.jpg` tiene etiqueta real `mountain`; el modelo predijo `glacier` con 0,8283. La imagen contiene picos nevados dentro de un mural/cartel; la semejanza visual con glaciares ayuda a explicar el error y muestra que una puntuación softmax alta no garantiza que la clase sea correcta.
- Interpretación: el modelo generaliza alrededor del 91% en datos no usados para entrenar. La confusión entre glacier/mountain y buildings/street es coherente con similitudes visuales entre categorías. Forest es la clase más distinguible en esta prueba.
- Para preservar `seg_test` como evaluación final independiente, el modelo se conserva como candidato final sin ajustar hiperparámetros a partir de estos errores.
- Archivos generados: `resultados/metricas_prueba.txt`, `resultados/matriz_confusion_prueba.png` y `resultados/predicciones_prueba.csv`.

## Predicción de demostración en seg_pred

- Archivo seleccionado de forma determinista: `archive/seg_pred/seg_pred/10004.jpg`.
- Clase predicha tras unificar el redimensionamiento bilineal con el pipeline de entrenamiento: `buildings`, puntuación softmax 0,6060 (60,60%); segunda opción `street`, 0,3940 (39,40%).
- Inspección visual: la imagen muestra una calle urbana con edificios, árboles y una torre; la predicción `buildings` resulta razonable; la puntuación alta de `street` muestra la ambigüedad entre fachadas urbanas y una escena de calle. Como `seg_pred` no tiene etiquetas, no se puede confirmar si es correcta.
- Evidencia: `resultados/prediccion_muestra_seg_pred.png` y `resultados/prediccion_muestra_seg_pred.txt`.

## Aplicación web

Se creó `app/app.py` con Streamlit. Acepta JPG/JPEG/PNG, convierte a RGB, redimensiona a 160 × 160 y usa el modelo guardado, que incluye la normalización de MobileNetV2. Presenta la clase, puntuación softmax y puntuaciones de las seis clases. La interfaz indica que la puntuación es la salida del modelo y no necesariamente una probabilidad calibrada.


- Verificación de la aplicación: Streamlit 1.66.1 instalado en .venv; el servidor inició y http://localhost:8501 respondió HTTP 200. El servidor de comprobación se cerró al terminar.


## Preparación para GitHub y despliegue

- Se guardó `modelos/clases.json` en el orden de salidas del modelo: buildings, forest, glacier, mountain, sea, street. La aplicación y los scripts de inferencia leen ese archivo para evitar desalinear nombres y neuronas.
- La normalización de MobileNetV2 sigue integrada en el modelo; la app convierte a RGB y redimensiona a 160 × 160, igual que el pipeline de inferencia.
- Los scripts de datos usan `INTEL_DATASET_DIR` o, por defecto, `archive` junto al proyecto. El dataset se excluye del repositorio.
- Se separaron dependencias mínimas de la app (`requirements.txt`) y paquetes extra de entrenamiento (`requirements-training.txt`); se fijaron TensorFlow 2.21.0 y Streamlit 1.66.1.
- Se añadió `.gitignore` para excluir entornos virtuales, datos, caches y secretos. Las rutas guardadas en el CSV de evaluación se dejaron relativas para evitar publicar rutas personales de este equipo.
- Se difirió la importación de TensorFlow hasta la primera imagen, para que la pantalla inicial no tenga que esperar el inicio de TensorFlow.
- Despliegue en Streamlit Community Cloud realizado el 3 de noviembre de 2026 desde `main`, con archivo principal `app/app.py` y Python 3.13.18. Primer intento: error de instalación porque Streamlit seleccionó Python 3.14, para el que TensorFlow 2.21.0 no tenía rueda compatible; se corrigió en la configuración de la app a Python 3.13 y las dependencias se instalaron correctamente. La página publicada abrió con el clasificador. La prueba de carga e inferencia de una imagen en la app remota sigue pendiente, así que todavía no se afirma que el ciclo remoto completo esté validado. URL: https://clasificacion-escenas-intel-d5sra3kmczl6scjagmh4ib.streamlit.app/.

- Verificación del artefacto ejecutada con `codigo/05_verificar_modelo.py`: el `.keras` cargó, la entrada fue `(None, 160, 160, 3)`, se encontraron seis clases en el orden esperado y la inferencia de comprobación produjo seis salidas finitas.

## Auditoría previa al informe (3 de noviembre de 2026)

- Requisitos con evidencia disponible: análisis de seis clases y conteos, partición reproducible 80/20 para entrenamiento/validación, conjunto de prueba reservado, redimensionamiento 150 × 150 a 160 × 160, aumento de datos solo en entrenamiento, normalización MobileNetV2 integrada, extracción de características con base congelada, regularización con Dropout y callbacks, métricas multiclase, matriz de confusión, curvas, predicción sobre `seg_pred`, artefacto `.keras` y aplicación web desplegada.
- Resultados registrados: mejor checkpoint por `val_loss` en época 14 (0,2199; `val_accuracy` 91,91 %); `seg_test`: accuracy 91,37 %, precision macro 91,55 %, recall macro 91,57 % y F1 macro 91,54 %.
- Falta un dato que no puede reconstruirse con certeza: el tiempo total real del entrenamiento. El script no guardó duración y ni el historial CSV ni el resumen del modelo incluyen ese dato. No se debe inventar ni inferir a partir de las fechas de archivos. Para reportar una duración exacta hay que volver a ejecutar el entrenamiento con medición explícita, o declarar honestamente que no quedó registrada.
- La justificación de CNN frente a ANN, transferencia de aprendizaje y tamaño 160 × 160 quedó documentada de forma retrospectiva abajo. En el informe hay que distinguir esta justificación técnica de lo que quedó registrado durante el experimento. El historial confirma 15 épocas ejecutadas y el checkpoint final elegido por `val_loss` corresponde a la época 14.
- No se ha probado aún una imagen cargada en la versión remota. La evaluación local en `seg_test` y la predicción local en `seg_pred` sí tienen resultados guardados; la app remota debe recibir una imagen de prueba para validar inferencia de punta a punta.
- La interfaz local se actualizó al español y ahora explica qué significa la clase elegida, muestra las seis puntuaciones y advierte cuando las dos primeras están próximas. Estas modificaciones son solo locales y aún no se han subido a GitHub ni a Streamlit Cloud.
## Decisiones de modelado documentadas para el informe

- **CNN frente a ANN:** las imágenes contienen bordes, texturas, formas y relaciones espaciales locales. Una ANN que aplane la imagen de 150 × 150 × 3 partiría de 67.500 valores por imagen y perdería la estructura espacial explícita; las convoluciones reutilizan filtros locales y reconocen patrones aunque cambien de posición.
- **Transfer Learning:** se usó MobileNetV2 con pesos de ImageNet como extractor de características, congelando la base y entrenando solo la cabeza de seis clases. Esta estrategia reutiliza filtros visuales generales y deja 7.686 parámetros entrenables frente a 2.265.670 parámetros totales; no se realizó fine-tuning.
- **Tamaño 160 × 160:** el tamaño original de 150 × 150 se redimensionó a la entrada de 160 × 160 usada por el experimento. Es una resolución admitida por MobileNetV2 y mantiene un costo inferior a 224 × 224; el resumen guardado confirma una salida espacial de 5 × 5 antes del pooling global. Esta explicación se añadió al revisar el código y debe presentarse como justificación del diseño, no como una nota contemporánea al entrenamiento si no fue documentada en ese momento.
- El aumento aleatorio se aplica solo durante entrenamiento: volteo horizontal, rotación 0,05 y zoom 0,10. MobileNetV2 recibe los valores transformados de 0..255 a -1..1 mediante una capa dentro del propio modelo. La inferencia aplica RGB y redimensionamiento bilineal a 160 × 160.
- Se añadieron mediciones con `time.perf_counter()` al script de entrenamiento. En una próxima ejecución se guardará en `resumen_modelo.txt` cuánto duró la llamada `model.fit`; el experimento ya ejecutado no contiene una medición recuperable.
- Verificación manual de la nueva interfaz local: se cargó `seg_pred/seg_pred/10004.jpg`; la app mostró **Edificios 60,6 %** y **Calle 39,4 %**, con texto descriptivo y las seis clases traducidas. Coincide con la predicción registrada por el script local. No se ha subido esta interfaz actualizada al repositorio ni a Streamlit Cloud.
- La explicación bajo la clase predicha describe el significado de la categoría y compara las dos puntuaciones mayores; no es una explicación de píxeles o regiones (no se implementó Grad-CAM ni otro método XAI).

- Cambio local del 3 de noviembre de 2026: la aplicación ofrece dos métodos de entrada, subir un JPG/JPEG/PNG o capturar una imagen con `st.camera_input`. La captura se carga en la app y usa la misma conversión RGB, corrección de orientación EXIF y redimensión a 160 × 160 que el flujo de inferencia. En celular, el navegador puede pedir permiso de cámara cuando se selecciona esa opción. Cambio pendiente de revisión; no se publicó en GitHub ni en Streamlit Cloud.