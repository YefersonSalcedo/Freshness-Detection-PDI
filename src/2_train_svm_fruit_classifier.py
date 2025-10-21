import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import joblib

# =============================================================
# Entrena un modelo SVM con las características HOG+LBP+LAB
# y guarda el modelo entrenado como modelo_svm_frutas.pkl
# =============================================================

RutaProyecto = os.getcwd()

X = np.load(os.path.join(RutaProyecto, "X_train.npy"))
Y = np.load(os.path.join(RutaProyecto, "Y_train.npy"))
clases = np.load(os.path.join(RutaProyecto, "clases.npy"))

print("Dataset cargado:")
print("X:", X.shape)
print("Y:", Y.shape)
print("Clases:", clases)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Entrenando modelo (Esto demora)...")
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nEvaluación del modelo:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision (macro):", metrics.precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", metrics.recall_score(y_test, y_pred, average='macro'))

joblib.dump(clf, os.path.join(RutaProyecto, "modelo_svm_frutas.pkl"))
print("\nModelo guardado con éxito como 'modelo_svm_frutas.pkl'")