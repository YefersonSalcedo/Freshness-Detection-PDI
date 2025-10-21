import os
import cv2 as cv
import numpy as np
from feature_extractor import extract_features

# =========================================================================
# Extrae las características HOG + LBP + LAB de todas las imágenes
# del dataset "Unified_Dataset" y las guarda como X_train.npy / Y_train.npy
# =========================================================================

RutaProyecto = os.getcwd()
RutaDataset = os.path.join(RutaProyecto, "Unified_Dataset")

X, Y, clases = [], [], []

for fruit in sorted(os.listdir(RutaDataset)):
    ruta_fruit = os.path.join(RutaDataset, fruit)
    if not os.path.isdir(ruta_fruit):
        continue

    for condition in sorted(os.listdir(ruta_fruit)):  # fresh / rotten
        ruta_clase = os.path.join(ruta_fruit, condition)
        if not os.path.isdir(ruta_clase):
            continue

        clase_nombre = f"{fruit}_{condition}"
        clases.append(clase_nombre)
        idx = len(clases) - 1
        print(f"Procesando clase: {clase_nombre} ({idx})")

        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith((".jpg", ".png", ".jpeg")):
                ruta_img = os.path.join(ruta_clase, archivo)
                img = cv.imread(ruta_img, cv.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv.resize(img, (224, 224))
                X.append(extract_features(img))
                Y.append(idx)

X, Y = np.array(X), np.array(Y)

np.save(os.path.join(RutaProyecto, "X_train.npy"), X)
np.save(os.path.join(RutaProyecto, "Y_train.npy"), Y)
np.save(os.path.join(RutaProyecto, "clases.npy"), np.array(clases))

print("Extracción completada")
print("Clases detectadas:", clases)
print("X shape:", X.shape)
print("Y shape:", Y.shape)
