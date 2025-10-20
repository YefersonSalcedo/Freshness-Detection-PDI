# Freshness Detection – Procesamiento Digital de Imágenes (PDI)

Este proyecto implementa un sistema de **detección de frescura de frutas** mediante técnicas de **Procesamiento Digital de Imágenes (PDI)** y **aprendizaje automático (Machine Learning)**.  
El objetivo es clasificar frutas como **frescas o podridas** a partir de imágenes, automatizando el proceso de inspección visual con un enfoque reproducible, modular y eficiente.

El modelo utiliza imágenes de frutas en dos condiciones —*fresh* y *rotten*—, procesadas para extraer características visuales que representan su textura, color y forma. Estas características alimentan un **clasificador SVM (Support Vector Machine)**, capaz de determinar el estado de frescura de cada fruta.

**Fuente del dataset:**  
El conjunto de datos se obtuvo del dataset público [**Fruit Quality** en Kaggle](https://www.kaggle.com/datasets/zlatan599/fruitquality1), que incluye imágenes de frutas frescas y podridas como manzanas, plátanos, naranjas, uvas y fresas, entre otras.

---

## Metodología y Flujo de Ejecución

El proyecto está compuesto por **cuatro módulos principales**, que forman un flujo de procesamiento completo: desde la extracción de características hasta la predicción final.

---

### `feature_extractor.py` — Extracción de Características  
Define la función `extract_features(img)`, que convierte cada imagen en un vector de características combinando:

- **HOG (Histogram of Oriented Gradients):** captura la forma y contornos de la fruta.  
- **LBP (Local Binary Patterns):** analiza texturas locales como rugosidades o manchas.  
- **Color LAB (media y desviación estándar):** mide los tonos y variaciones cromáticas asociadas a la madurez o descomposición.

Cada imagen se representa mediante un vector que resume su **textura, color y estructura**.

---

### `1_extract_features_fruit_dataset.py` — Construcción del Dataset  
Este script recorre la carpeta `Unified_Dataset/`, donde las imágenes están organizadas por fruta y condición (por ejemplo, `apple/fresh`, `apple/rotten`).  
Para cada imagen:

1. Se redimensiona a 224×224 píxeles.  
2. Se extraen las características usando `extract_features(img)`.  
3. Se almacenan los resultados en tres archivos:
   - `X_train.npy` → matriz de características  
   - `Y_train.npy` → etiquetas de clase  
   - `clases.npy` → nombres de clases detectadas

Así se genera el **dataset de entrenamiento** para el modelo.

---

### `2_train_svm_fruit_classifier.py` — Entrenamiento del Modelo  
Con los datos procesados, este módulo:

1. Carga `X_train.npy`, `Y_train.npy` y `clases.npy`.  
2. Divide los datos en conjuntos de entrenamiento y prueba (80/20).  
3. Entrena un **SVM lineal** para clasificar las frutas.  
4. Evalúa el rendimiento con métricas como:
   - *Accuracy (precisión global)*  
   - *Precision (macro)*  
   - *Recall (macro)*  
5. Guarda el modelo entrenado en `modelo_svm_frutas.pkl`.

---

### `3_test_fruit_classifier.py` — Clasificación de Nuevas Imágenes  
Permite realizar pruebas interactivas con imágenes nuevas:

1. Carga el modelo `modelo_svm_frutas.pkl` y las clases `clases.npy`.  
2. Abre un cuadro de diálogo (Tkinter) para seleccionar una imagen.  
3. Extrae sus características con `extract_features(img)`.  
4. Predice su clase (*fruit_fresh* o *fruit_rotten*) y muestra:
   - La clase predicha  
   - El nivel de confianza (%)  
   - La imagen con la etiqueta superpuesta

Este módulo demuestra el uso práctico del sistema entrenado.

---

### Nota importante:
Los archivos de datos generados (`X_train.npy`, `Y_train.npy`, `clases.npy`) y el modelo entrenado (`modelo_svm_frutas.pkl`) no se incluyen en este repositorio debido a su tamaño.
Para ejecutarlo correctamente, debes:

1. Descargar el [dataset original desde Kaggle](https://www.kaggle.com/datasets/zlatan599/fruitquality1).
2. Ejecutar el script `1_extract_features_fruit_dataset.py` para generar los archivos `.npy`.
3. Entrenar el modelo con `2_train_svm_fruit_classifier.py` para obtener el archivo `.pkl`.

