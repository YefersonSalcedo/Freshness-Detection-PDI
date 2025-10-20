import os
import cv2 as cv
import numpy as np
import joblib
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from feature_extractor import extract_features

# =============================================================
# Permite cargar una imagen manualmente y clasificarla
# como fruta fresca o podrida con el modelo entrenado.
# =============================================================

RutaProyecto = os.getcwd()
modelo_path = os.path.join(RutaProyecto, "modelo_svm_frutas.pkl")
clases_path = os.path.join(RutaProyecto, "clases.npy")

if not os.path.exists(modelo_path) or not os.path.exists(clases_path):
    raise FileNotFoundError("No se encontró el modelo o las clases. Ejecuta primero el entrenamiento.")

clf = joblib.load(modelo_path)
clases = np.load(clases_path)

# ----------------------------------------------------------
# Seleccionar imagen manualmente
# ----------------------------------------------------------
Tk().withdraw()  # Oculta la ventana principal de Tkinter
RutaImagen = askopenfilename(
    title="Selecciona una imagen de fruta",
    filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg")]
)

if not RutaImagen:
    raise ValueError("No seleccionaste ninguna imagen.")

img = cv.imread(RutaImagen)
if img is None:
    raise ValueError(f"No se pudo cargar la imagen: {RutaImagen}")

# ----------------------------------------------------------
# Procesar y predecir
# ----------------------------------------------------------
features = extract_features(img).reshape(1, -1)
pred = clf.predict(features)[0]
probs = clf.predict_proba(features)[0]

# ----------------------------------------------------------
# Mostrar resultado
# ----------------------------------------------------------
print("\nPredicción completada")
print("----------------------------")
print(f"Imagen: {os.path.basename(RutaImagen)}")
print(f"Clase predicha: {clases[pred]}")
print(f"Confianza: {np.max(probs)*100:.2f}%")

# Mostrar imagen con etiqueta
label = f"{clases[pred]} ({np.max(probs)*100:.1f}%)"
cv.putText(img, label, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv.imshow("Resultado de Clasificación", cv.resize(img, (400,400)))
cv.waitKey(0)
cv.destroyAllWindows()