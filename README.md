# Proyecto IA: clasificación de escenas

Clasificador multiclase del conjunto **Intel Image Classification**. Predice seis categorías: `buildings`, `forest`, `glacier`, `mountain`, `sea` y `street`.

## Estructura

- `app/app.py`: aplicación web Streamlit.
- `codigo/`: exploración, preparación, entrenamiento, evaluación, predicción y verificación del modelo.
- `modelos/`: modelo `.keras` y `clases.json` en el orden de las salidas.
- `resultados/`: métricas, curvas, matriz de confusión y predicciones generadas.
- `requirements.txt`: dependencias mínimas para ejecutar/desplegar la aplicación.
- `requirements-training.txt`: dependencias adicionales para entrenar y usar notebooks.
- `bitacora_proyecto.md`: decisiones y resultados organizados para el informe CRISP-DM.

El dataset no se incluye en GitHub. Por defecto, los scripts lo buscan en `archive`, una carpeta hermana del proyecto. Si está en otra ubicación, define la variable `INTEL_DATASET_DIR` antes de ejecutar los scripts.

## Entorno local en Windows

En este equipo se entrenó con TensorFlow 2.21.0 en el entorno `.venv` existente. Para instalar las dependencias necesarias para los scripts de entrenamiento:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-training.txt
```

El entorno local usado aquí es Python 3.13.5. La guía de la asignatura menciona Python 3.10–3.12 como rango general. Para el despliegue hay que elegir en Streamlit Cloud una versión compatible con TensorFlow 2.21.0 y comprobar que la instalación termina correctamente; la página oficial de TensorFlow actualmente lista paquetes para Python 3.13.

## Flujo del proyecto

Desde la raíz `Proyecto_IA_Escenas` y con el entorno activo:

```powershell
python codigo/01_preparacion_datos.py
python codigo/02_entrenamiento_mobilenetv2.py
python codigo/03_evaluacion_final.py
python codigo/04_prediccion_muestra.py
python codigo/05_verificar_modelo.py
```

`seg_train` se separa en entrenamiento y validación (80/20); `seg_test` se reserva para evaluación final; `seg_pred` se usa para una demostración sin etiqueta real. El entrenamiento guarda el modelo y `modelos/clases.json`. El último script aplica la comprobación de carga e inferencia recomendada en la guía.

## Aplicación local

```powershell
python -m pip install -r requirements.txt
streamlit run app/app.py
```

La página recibe JPG/JPEG/PNG, convierte a RGB, redimensiona a 160 × 160 y usa la normalización MobileNetV2 integrada en el modelo. TensorFlow se carga al recibir la primera imagen y luego queda cacheado.

## Despliegue en Streamlit Community Cloud

Sube el código, `app/`, `modelos/`, `requirements.txt` y la documentación a GitHub. No subas `.venv`, el dataset ni secretos. En Streamlit Community Cloud selecciona el repositorio, rama y archivo principal `app/app.py`; fija explícitamente la versión de Python y revisa los logs de construcción. Prueba la URL publicada con una imagen y compara el preprocesamiento y la salida con la aplicación local.
