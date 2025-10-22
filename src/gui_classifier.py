import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2 as cv
import joblib
import numpy as np
from PIL import Image, ImageTk
from sklearn import metrics

from feature_extractor import extract_features


class GUIClassifier(tk.Tk):
    """Interfaz gráfica simple para cargar un modelo, seleccionar imágenes/carpeta
    y clasificar usando el extractor de características y el clasificador SVM guardado.
    """

    def __init__(self):
        super().__init__()
        self.title("Fruit Freshness Classifier")
        self.geometry("900x600")

        # Estado
        self.model = None
        self.classes = None
        self.model_path = None
        self.image_path = None

        # UI
        self._build_ui()

    def _build_ui(self):
        frm_controls = ttk.Frame(self)
        frm_controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        btn_load_model = ttk.Button(frm_controls, text="Cargar modelo (.pkl)", command=self.load_model)
        btn_load_model.grid(row=0, column=0, padx=4)

        btn_load_classes = ttk.Button(frm_controls, text="Cargar clases (.npy)", command=self.load_classes)
        btn_load_classes.grid(row=0, column=1, padx=4)

        btn_select_image = ttk.Button(frm_controls, text="Seleccionar imagen", command=self.select_image)
        btn_select_image.grid(row=0, column=2, padx=4)

        btn_select_folder = ttk.Button(frm_controls, text="Seleccionar carpeta", command=self.select_folder)
        btn_select_folder.grid(row=0, column=3, padx=4)

        btn_classify_image = ttk.Button(frm_controls, text="Clasificar imagen", command=self.classify_image_thread)
        btn_classify_image.grid(row=0, column=4, padx=4)

        btn_classify_folder = ttk.Button(frm_controls, text="Clasificar carpeta", command=self.classify_folder_thread)
        btn_classify_folder.grid(row=0, column=5, padx=4)

        # Panel principal dividido: imagen a la izquierda, logs a la derecha
        pan = ttk.Frame(self)
        pan.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Imagen
        frm_image = ttk.LabelFrame(pan, text="Imagen")
        frm_image.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Label(frm_image)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Resultados
        frm_results = ttk.LabelFrame(pan, text="Resultados / Log")
        frm_results.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        frm_results.config(width=380)

        self.lbl_model = ttk.Label(frm_results, text="Modelo: (no cargado)")
        self.lbl_model.pack(anchor=tk.W, padx=6, pady=(6,0))

        self.lbl_image = ttk.Label(frm_results, text="Imagen: (no seleccionada)")
        self.lbl_image.pack(anchor=tk.W, padx=6, pady=(0,6))

        self.txt = tk.Text(frm_results, width=50, height=30)
        self.txt.pack(padx=6, pady=6)

    # ---------- acciones UI ----------
    def load_model(self):
        path = filedialog.askopenfilename(title="Selecciona archivo de modelo (.pkl)", filetypes=[("Pickle", "*.pkl"), ("All files", "*")])
        if not path:
            return
        try:
            clf = joblib.load(path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
            return
        self.model = clf
        self.model_path = path
        self.lbl_model.config(text=f"Modelo: {os.path.basename(path)}")
        self.log(f"Modelo cargado: {path}")

    def load_classes(self):
        path = filedialog.askopenfilename(title="Selecciona archivo de clases (.npy)", filetypes=[("NumPy", "*.npy"), ("All files", "*")])
        if not path:
            return
        try:
            clases = np.load(path, allow_pickle=True)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar las clases:\n{e}")
            return
        self.classes = clases
        self.log(f"Clases cargadas: {path} -> {list(clases)}")

    def select_image(self):
        path = filedialog.askopenfilename(title="Selecciona imagen", filetypes=[("Imagen", "*.jpg *.jpeg *.png"), ("All files", "*")])
        if not path:
            return
        self.image_path = path
        self.lbl_image.config(text=f"Imagen: {os.path.basename(path)}")
        self.show_image(path)
        self.log(f"Imagen seleccionada: {path}")

    def select_folder(self):
        path = filedialog.askdirectory(title="Selecciona carpeta con imágenes o subcarpetas por clase")
        if not path:
            return
        # Guardamos la carpeta en self.image_path para uso por classify_folder
        self.image_path = path
        self.lbl_image.config(text=f"Carpeta: {os.path.basename(path)}")
        self.log(f"Carpeta seleccionada: {path}")

    def show_image(self, path, maxsize=(400, 400)):
        try:
            pil = Image.open(path).convert('RGB')
        except Exception:
            # fallback cv2
            img = cv.imread(path)
            if img is None:
                return
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            pil = Image.fromarray(img)
        pil.thumbnail(maxsize)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.config(image=self.photo)

    def log(self, text):
        self.txt.insert(tk.END, text + "\n")
        self.txt.see(tk.END)

    # ---------- clasificación (hilos para no bloquear UI) ----------
    def classify_image_thread(self):
        t = threading.Thread(target=self.classify_image)
        t.daemon = True
        t.start()

    def classify_folder_thread(self):
        t = threading.Thread(target=self.classify_folder)
        t.daemon = True
        t.start()

    def classify_image(self):
        if self.model is None:
            messagebox.showwarning("Modelo no cargado", "Carga un modelo antes de clasificar.")
            return
        if not self.image_path or not os.path.isfile(self.image_path):
            messagebox.showwarning("Imagen no seleccionada", "Selecciona una imagen para clasificar.")
            return

        img = cv.imread(self.image_path)
        if img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen con OpenCV.")
            return

        try:
            feats = extract_features(img).reshape(1, -1)
        except Exception as e:
            messagebox.showerror("Error", f"Error extrayendo características:\n{e}")
            return

        try:
            pred = self.model.predict(feats)[0]
            probs = None
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(feats)[0]
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción:\n{e}")
            return

        class_name = str(pred)
        conf_text = ""
        if self.classes is not None:
            try:
                class_name = self.classes[pred]
            except Exception:
                class_name = str(pred)

        if probs is not None:
            conf = np.max(probs)
            conf_text = f" (Confianza: {conf*100:.1f}%)"

        self.log(f"Resultado: {class_name}{conf_text}")
        # actualizar imagen (ya mostrada), y poner etiqueta arriba
        display_label = f"{class_name}{conf_text}"
        img_disp = cv.imread(self.image_path)
        if img_disp is not None:
            cv.putText(img_disp, display_label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            tmp_path = os.path.join(os.path.dirname(self.image_path), "__tmp_preview.jpg")
            cv.imwrite(tmp_path, img_disp)
            self.show_image(tmp_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def classify_folder(self):
        if self.model is None:
            messagebox.showwarning("Modelo no cargado", "Carga un modelo antes de clasificar.")
            return
        if not self.image_path or not os.path.isdir(self.image_path):
            messagebox.showwarning("Carpeta no seleccionada", "Selecciona una carpeta para clasificar.")
            return

        # recorrer la carpeta: si contiene subcarpetas, usarlas como etiquetas verdaderas
        y_true = []
        y_pred = []
        classes_found = set()
        files = []
        for root, dirs, filenames in os.walk(self.image_path):
            for f in filenames:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    files.append(os.path.join(root, f))

        if not files:
            messagebox.showinfo("Ninguna imagen", "No se encontraron imágenes en la carpeta seleccionada.")
            return

        self.log(f"Clasificando {len(files)} imágenes...")

        for fp in files:
            img = cv.imread(fp)
            if img is None:
                continue
            try:
                feats = extract_features(img).reshape(1, -1)
                pred = self.model.predict(feats)[0]
            except Exception as e:
                self.log(f"Error procesando {fp}: {e}")
                continue

            y_pred.append(pred)

            # intento inferir etiqueta verdadera si la estructura es .../class_name/imagen.jpg
            parent = os.path.basename(os.path.dirname(fp))
            classes_found.add(parent)
            if self.classes is not None:
                # si las clases cargadas contienen parent string, convertir a indice
                try:
                    # buscar índice donde self.classes == parent
                    idx = int(np.where(self.classes == parent)[0])
                    y_true.append(idx)
                except Exception:
                    # no se puede mapear -> no añadimos etiqueta verdadera
                    y_true.append(None)
            else:
                y_true.append(parent)

        # si y_true contiene indices válidos (int) podemos calcular métricas
        valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if isinstance(t, (int, np.integer))]
        if valid_pairs:
            t_vals, p_vals = zip(*valid_pairs)
            acc = metrics.accuracy_score(t_vals, p_vals)
            report = metrics.classification_report(t_vals, p_vals, target_names=[str(c) for c in self.classes], zero_division=0)
            cm = metrics.confusion_matrix(t_vals, p_vals)
            self.log(f"Accuracy: {acc:.4f}")
            self.log("Classification report:\n" + report)
            self.log("Confusion matrix:\n" + np.array2string(cm))
        else:
            # mostrar conteo de predicciones por clase
            unique, counts = np.unique(y_pred, return_counts=True)
            summary = "Conteo predicciones:\n"
            for u, c in zip(unique, counts):
                name = str(u)
                if self.classes is not None:
                    try:
                        name = str(self.classes[u])
                    except Exception:
                        pass
                summary += f"  {name}: {c}\n"
            self.log(summary)


if __name__ == '__main__':
    app = GUIClassifier()
    app.mainloop()
