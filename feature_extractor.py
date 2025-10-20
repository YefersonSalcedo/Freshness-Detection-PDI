import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern

# ================================================================
# Función común para extraer características HOG + LBP + Color LAB
# ================================================================

def extract_features(img):
    # ================================================================
    # =============ASEGURAR 3 CANALES (BGR) ==========================
    # ================================================================
    # Esto es necesario porque para convertir a LAB se requieren 3 canales de color.

    if img is None:  # Si la imagen no se cargó correctamente (None), lanza un error
        raise ValueError("Imagen vacía o no cargada correctamente.")
    if len(img.shape) == 2:  # Si la imagen está en escala de grises (solo tiene 2 dimensiones),
         img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) # se convierte a BGR duplicando los canales.
    elif img.shape[2] == 4:  # Si la imagen tiene 4 canales (por ejemplo, formato PNG con transparencia)
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR) # se elimina el canal alfa (A) para dejarla en formato BGR de 3 canales.

    # ================================================================
    # =============PREPROCESAMIENTO DE LA IMAGEN======================
    # ================================================================
    img = cv.resize(img, (224,224))

    # Convierte la imagen del espacio de color BGR al espacio LAB.
    # LAB separa la información de luminosidad (L) y color (A,B),
    # lo que facilita mejorar el contraste sin alterar el color.
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(img_lab) #Divide los tres canales: L (luminancia), A (verde-rojo), B (azul-amarillo)

    # Ecualiza el histograma del canal L para mejorar el contraste general.
    # Esto resalta los detalles de textura (arrugas, manchas, moho, etc.)
    l = cv.equalizeHist(l)
    img_lab = cv.merge((l,a,b)) #Combina nuevamente los canales L, A y B en una sola imagen LAB.

    # Convierte la imagen LAB a BGR (para compatibilidad)
    # y luego a escala de grises, para aplicar HOG y LBP más adelante.
    img_gray = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)


    # ================================================================
    # ==========HOG (Histogram of Oriented Gradients)=================
    # ================================================================

    # HOG se usa para capturar la forma y los bordes de los objetos.
    # Parámetros:
    # - winSize = (224,224): tamaño total de la imagen
    # - blockSize = (16,16): tamaño de los bloques de análisis
    # - blockStride = (8,8): paso entre bloques
    # - cellSize = (8,8): tamaño de las celdas dentro de cada bloque
    # - nbins = 9: número de orientaciones de gradiente
    hog = cv.HOGDescriptor((224,224), (16,16), (8,8), (8,8), 9)

    # Calcula el descriptor HOG sobre la imagen en escala de grises
    # y lo convierte en un vector unidimensional (flatten).
    hog_desc = hog.compute(img_gray).flatten()



    # ================================================================
    # =============LBP (Local Binary Patterns)========================
    # ================================================================

    # LBP analiza la textura local comparando cada píxel con sus vecinos.
    # Detecta microtexturas (rugosidad, manchas, moho, etc.)
    radius = 3              # Radio de vecindad
    n_points = 8 * radius   # Número de puntos vecinos (24)

    # Calcula la imagen de patrones binarios locales (LBP)
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')

    # Genera un histograma de los valores LBP obtenidos.
    # Este histograma representa cuántas veces aparecen ciertas texturas.
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points+2))

    # Convierte a flotante y normaliza el histograma
    # (para que la suma total sea 1). Así no depende del tamaño de la imagen.
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-6)



    # ================================================================
    # =======COLOR LAB (estadísticas del espacio LAB)=================
    # ================================================================

    # Calcula la media (color promedio) de los canales L, A y B
    # Esto capta el color global (una fruta podrida tiende tener un tono más oscuro)
    mean_color = np.mean(img_lab, axis=(0,1))

    # Calcula la desviación estándar del color, que mide la variación cromática.
    # Por ejemplo, una fruta podrida puede tener zonas de color irregular.
    std_color = np.std(img_lab, axis=(0,1))


    # ================================================================
    # ===========CONCATENAR TODAS LAS CARACTERISTICAS=================
    # ================================================================

    # Combina todos los vectores en un solo arreglo unidimensional:
    # [ HOG features | LBP histogram | LAB mean | LAB std ]
    # Esto forma el vector de características completo que representará a la imagen.
    features = np.concatenate((hog_desc, hist_lbp, mean_color, std_color))

    # Devuelve el vector de características resultante.
    return features