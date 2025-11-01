import cv2
import pytesseract
import numpy as np
import re
import os
import sys
from glob import glob

# ----------------------------
# CONFIGURACIÓN GLOBAL
# ----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------------
# FUNCIONES DE PROCESAMIENTO
# ----------------------------

def preprocess_image(path, debug=False):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.medianBlur(gray, 3)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(
        thresh_otsu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    if debug:
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Binarizada", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return thresh

def ocr_and_confidence(img, lang="spa", psm=6, oem=1):
    config = f'--oem {oem} --psm {psm} -l {lang}'
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(img, config=config)

    # Limpieza avanzada
    text = re.sub(r'[^\w\s.,;:¡!¿?()\-–—%€$áéíóúÁÉÍÓÚñÑ\n]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = text.strip()

    # Calcular confianza media real
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
    binaria = preprocess_image(img_path, debug=debug)
    texto, conf, data = ocr_and_confidence(binaria, lang=lang, psm=6, oem=1)
    return texto, conf

# ----------------------------
# PROCESAMIENTO DE CARPETA
# ----------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python text_extractor_from_folder.py <carpeta_imagenes> [--debug]")
        sys.exit(1)

    carpeta = sys.argv[1]
    debug_flag = "--debug" in sys.argv

    # Obtener todas las imágenes jpg, jpeg, png
    extensiones = ("*.jpg", "*.jpeg", "*.png")
    imagenes = []
    for ext in extensiones:
        imagenes.extend(glob(os.path.join(carpeta, ext)))
    imagenes.sort()  # orden alfabético

    if not imagenes:
        print("No se encontraron imágenes en la carpeta.")
        sys.exit(1)

    print(f"Se encontraron {len(imagenes)} imágenes. Procesando...")

    # Procesar cada imagen y guardar texto
    for idx, img_path in enumerate(imagenes, start=1):
        texto, conf = extraer_texto_imagen(img_path, debug=debug_flag)
        salida_txt = f"text{idx}.txt"
        with open(salida_txt, "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"{img_path} -> {salida_txt} | Confianza: {conf}%")

    print("Procesamiento completado.")
