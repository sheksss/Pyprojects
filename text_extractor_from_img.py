"""
Script de extracción automática de texto desde imágenes mediante OCR (Tesseract).

Este programa recorre una carpeta con imágenes (formatos JPG, JPEG y PNG),
aplica un preprocesamiento para mejorar la calidad del texto (aumentar contraste,
eliminar ruido, binarizar, etc.) y luego utiliza Tesseract OCR para reconocer
el texto y guardarlo en archivos .txt separados.

Uso:
python text_extractor_from_folder.py <carpeta_imagenes> [--debug]

Ejemplo:
python text_extractor_from_folder.py "C:\\Users\\Jon\\Pictures\\documentos" --debug

Dependencias:
- OpenCV (cv2)
- pytesseract
- numpy

Salida:
Se generan archivos text1.txt, text2.txt, ... con el texto reconocido
y se muestra en consola la confianza media del OCR por imagen.
"""

# ----------------------------
# IMPORTACIÓN DE LIBRERÍAS
# ----------------------------
import cv2              # Procesamiento de imágenes
import pytesseract      # Interfaz con Tesseract OCR
import numpy as np      # Cálculos numéricos
import re               # Expresiones regulares para limpiar texto
import os
import sys
from glob import glob   # Para buscar archivos con patrones (*.jpg, *.png)


# ----------------------------
# FUNCIONES DE PROCESAMIENTO
# ----------------------------

def preprocess_image(path, debug=False):
    """
    Preprocesa una imagen para mejorar los resultados del OCR.

    Convierte la imagen a escala de grises, aumenta su resolución,
    mejora el contraste mediante CLAHE, elimina ruido con un filtro de mediana
    y aplica binarización (Otsu y adaptativa) para obtener una imagen
    más legible para Tesseract.

    @param path  Ruta del archivo de imagen que se desea procesar.
    @param debug Si es True, muestra las imágenes intermedias (grises y binarizadas).

    @return Imagen binarizada (numpy.ndarray) lista para aplicar OCR.

    @raises FileNotFoundError Si no se puede abrir la imagen indicada.
    """

    # Leer la imagen desde la ruta proporcionada
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aumentar tamaño para mejorar el reconocimiento de caracteres pequeños
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

    # Mejorar contraste usando ecualización adaptativa (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Reducir ruido preservando bordes (filtro de mediana)
    blur = cv2.medianBlur(gray, 3)

    # Binarización automática de Otsu (separa fondo y texto)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Binarización adaptativa (útil si hay zonas con distinta iluminación)
    thresh = cv2.adaptiveThreshold(
        thresh_otsu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Mostrar resultados intermedios si se activa el modo debug
    if debug:
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Binarizada", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return thresh


def ocr_and_confidence(img, lang="spa", psm=6, oem=1):
    """
    Aplica Tesseract OCR a una imagen y calcula la confianza media del reconocimiento.

    Extrae texto y datos detallados palabra por palabra, limpia los resultados
    y devuelve el texto final junto a la confianza media del motor OCR.

    @param img   Imagen binarizada (numpy.ndarray) a procesar.
    @param lang  Código de idioma (por defecto "spa" para español).
    @param psm   Page Segmentation Mode (por defecto 6: bloque de texto uniforme).
    @param oem   OCR Engine Mode (por defecto 1: motor LSTM moderno).

    @return text       Texto reconocido, limpio y normalizado.
    @return conf_mean  Confianza media del OCR (float, 0–100).
    @return data       Diccionario con datos detallados por palabra.
    """

    # Configuración de Tesseract OCR
    config = f'--oem {oem} --psm {psm} -l {lang}'

    # Extraer datos palabra por palabra (posición, texto, confianza, etc.)
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

    # Extraer texto completo en una sola cadena
    text = pytesseract.image_to_string(img, config=config)

    # Limpieza avanzada del texto (elimina caracteres raros, dobles espacios, etc.)
    text = re.sub(r'[^\w\s.,;:¡!¿?()\-–—%€$áéíóúÁÉÍÓÚñÑ\n]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = text.strip()

    # Calcular confianza media (ignorando valores -1 y vacíos)
    confs = []
    for c in data.get("conf", []):
        try:
            c_str = str(c).strip()
            if c_str != "" and c_str != "-1":
                confs.append(int(float(c_str)))
        except (ValueError, TypeError):
            continue

    conf_mean = round(float(np.mean(confs)), 2) if confs else 0.0

    return text, conf_mean, data


def extraer_texto_imagen(img_path, lang="spa", debug=False):
    """
    Extrae texto de una imagen completa aplicando preprocesamiento y OCR.

    Combina las funciones preprocess_image() y ocr_and_confidence()
    para obtener el texto y su nivel de confianza en un solo paso.

    @param img_path Ruta del archivo de imagen.
    @param lang     Idioma del OCR (por defecto "spa").
    @param debug    Si es True, muestra imágenes intermedias de depuración.

    @return texto Texto extraído de la imagen (str).
    @return conf  Confianza media del OCR (float).
    """

    # Paso 1: Preprocesar la imagen para mejorar legibilidad del texto
    binaria = preprocess_image(img_path, debug=debug)

    # Paso 2: Aplicar OCR sobre la imagen binarizada
    texto, conf, data = ocr_and_confidence(binaria, lang=lang, psm=6, oem=1)

    return texto, conf


# ----------------------------
# PROCESAMIENTO DE CARPETA
# ----------------------------
if __name__ == "__main__":
    """
    Programa principal.

    Recorre una carpeta con imágenes (.jpg, .jpeg, .png), aplica
    preprocesamiento y OCR a cada una, y guarda el texto reconocido
    en archivos de salida (text1.txt, text2.txt, ...).

    Uso:
    python text_extractor_from_folder.py <carpeta_imagenes> [--debug]

    @param carpeta_imagenes Ruta de la carpeta que contiene las imágenes a procesar.
    @param --debug          (opcional) Muestra las imágenes intermedias.

    @output Genera un archivo de texto por imagen procesada.
    """

    # Validar argumentos
    if len(sys.argv) < 2:
        print("Uso: python text_extractor_from_folder.py <carpeta_imagenes> [--debug]")
        sys.exit(1)

    carpeta = sys.argv[1]
    debug_flag = "--debug" in sys.argv

    # Buscar imágenes compatibles en la carpeta (jpg, jpeg, png)
    extensiones = ("*.jpg", "*.jpeg", "*.png")
    imagenes = []
    for ext in extensiones:
        imagenes.extend(glob(os.path.join(carpeta, ext)))
    imagenes.sort()  # Orden alfabético

    if not imagenes:
        print("No se encontraron imágenes en la carpeta.")
        sys.exit(1)

    print(f"Se encontraron {len(imagenes)} imágenes. Procesando...\n")

    # Procesar cada imagen y guardar el texto resultante
    for idx, img_path in enumerate(imagenes, start=1):
        texto, conf = extraer_texto_imagen(img_path, debug=debug_flag)

        # Crear archivo de salida con el texto reconocido
        salida_txt = f"texto{idx}.txt"
        with open(salida_txt, "w", encoding="utf-8") as f:
            f.write(texto)

        print(f"[{idx}/{len(imagenes)}] {os.path.basename(img_path)} -> {salida_txt} | Confianza media: {conf}%")

    print("\nProcesamiento completado.")
