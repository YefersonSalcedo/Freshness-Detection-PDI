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
- **Color LAB (media y desviación estándar):** mide los tonos y variaciones cromáticas asociadas a la madurez o descomposición.

Cada imagen se representa mediante un vector que resume su **textura, color y estructura**.

---

### `extract_features_fruit_dataset.py` — Construcción del Dataset  
Este script recorre la carpeta `Unified_Dataset/`, donde las imágenes están organizadas por fruta y condición (por ejemplo, `apple/fresh`, `apple/rotten`).  
Para cada imagen:

1. Se redimensiona a 112×112 píxeles.  
2. Se extraen las características usando `extract_features(img)`.  
3. Se almacenan los resultados en tres archivos:
   - `X_train.npy` → matriz de características  
   - `Y_train.npy` → etiquetas de clase  
   - `clases.npy` → nombres de clases detectadas

Así se genera el **dataset de entrenamiento** para el modelo.

---

### `train_svm_fruit_classifier.py` — Entrenamiento del Modelo  
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

### `gui_classifie.py` — Interfaz Gráfica para Clasificación de Imágenes  
Este módulo proporciona una interfaz interactiva que permite clasificar imágenes individuales o carpetas completas utilizando el modelo previamente entrenado.

#### Funcionalidades principales:
1. **Carga del modelo y clases:**
   Permite importar el modelo entrenado `modelo_svm_frutas.pkl` y las clases `clases.npy` mediante cuadros de diálogo.

2. **Selección de imágenes o carpetas:**
   * Puedes seleccionar una **imagen individual** para prueba.
   * O una **carpeta completa** (útil para evaluar conjuntos de imágenes o datasets).

4. **Extracción de características:**
   Cada imagen seleccionada es procesada con la función `extract_features(img)` definida en `feature_extractor.py`, que convierte la imagen en un vector de características compatible con el modelo SVM.

5. **Clasificación:**
   * El modelo predice la clase de la imagen (por ejemplo, *fruit_fresh* o *fruit_rotten*).
   * Si el clasificador soporta probabilidades, también muestra el **nivel de confianza (%)** de la predicción.
   * La imagen se muestra en la ventana con la **etiqueta superpuesta** sobre ella.

6. **Evaluación por carpetas:**
   Si se selecciona una carpeta con subcarpetas por clase, el sistema muestra un conteo de imágenes clasificadas por clase.

7. **Interfaz amigable y no bloqueante:**
   Todas las tareas de clasificación se ejecutan en **hilos independientes**, manteniendo la interfaz fluida.
   Incluye registro en tiempo real de las acciones y resultados en un panel de texto lateral.

**Uso típico:**

1. Ejecutar `gui_classifier.py`.
2. Cargar el modelo y las clases.
3. Seleccionar una imagen o carpeta.
4. Presionar “Clasificar imagen” o “Clasificar carpeta”.

---

### Archivos de Investigación

El proyecto incluye un módulo adicional de **investigación sobre filtros**. Este archivo contiene experimentos y pruebas con diferentes técnicas de filtrado (como filtros gaussianos, logaritmicos, media, entre otros) que fueron explorados durante el desarrollo del proyecto.

**Nota:** Este módulo es únicamente de carácter exploratorio y **no está integrado** en el flujo principal de clasificación. Su propósito es documentar las técnicas investigadas y servir como referencia para futuras mejoras o extensiones del sistema.

---

### Nota importante

Los archivos generados durante el entrenamiento (`X_train.npy`, `Y_train.npy`, `clases.npy`) y el modelo final (`modelo_svm_frutas.pkl`) **no están incluidos en este repositorio** debido a su tamaño.

Dependiendo de lo que desees hacer, sigue una de las siguientes opciones:

#### Si deseas **entrenar el modelo desde cero**:

1. Descarga el [dataset original desde Kaggle](https://www.kaggle.com/datasets/zlatan599/fruitquality1).
2. Ejecuta el script `extract_features_fruit_dataset.py` para generar los archivos `.npy` con las características y etiquetas.
3. Entrena el modelo utilizando `train_svm_fruit_classifier.py`, lo que generará el archivo `modelo_svm_frutas.pkl`.

#### Si solo deseas **probar la clasificación con la interfaz gráfica**:

Descarga directamente los archivos ya entrenados desde el siguiente enlace:

[**Modelo y clases — Google Drive**](https://drive.google.com/drive/folders/1lnS6QjRLefLl3hHp6Elh5Mjx-nhwLOky?usp=drive_link)

Luego, simplemente ejecuta `gui_classifier.py`, carga el modelo y las clases, y comienza a clasificar tus imágenes.

