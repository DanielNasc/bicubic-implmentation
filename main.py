import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

# Carrega a imagem em tons de cinza
img = cv2.imread("assets/cat.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None

# Aplica limiarização para facilitar a visualização
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
IMG_H, IMG_W = img.shape

# Função kernel cúbico
def cubic_kernel(x: float) -> float:
    x = abs(x)
    if x < 1:
        return 1.5 * x**3 - 2.5 * x**2 + 1
    elif x < 2:
        return -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2
    else:
        return 0.0

def get_pixel(img, x, y):
    h, w = img.shape
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return img[int(y), int(x)]

def bicubic_interpolate(img: np.ndarray, x: float, y: float) -> float:
    result = 0.0
    x_int = int(np.floor(x))
    y_int = int(np.floor(y))
    for m in range(-1, 3):
        for n in range(-1, 3):
            px = x_int + n
            py = y_int + m
            dx = x - px
            dy = y - py
            weight_x = cubic_kernel(dx)
            weight_y = cubic_kernel(dy)
            pixel = get_pixel(img, px, py)
            result += pixel * weight_x * weight_y
    return np.clip(result, 0, 255)

def apply_transform(img, transform_func, interpolator):
    h_out, w_out = transform_func(img.shape)
    output = np.zeros((h_out, w_out), dtype=np.uint8)

    for y_out in range(h_out):
        for x_out in range(w_out):
            x_in, y_in = transform_func((y_out, x_out), inverse=True)
            if 0 <= x_in < img.shape[1] and 0 <= y_in < img.shape[0]:
                if interpolator == "nearest":
                    value = get_pixel(img, int(round(x_in)), int(round(y_in)))
                elif interpolator == "bicubic":
                    value = bicubic_interpolate(img, x_in, y_in)
                output[y_out, x_out] = value
    return output

# Escalonamento
def scale_transform(shape_or_coords, s=2.0, inverse=False):
    if isinstance(shape_or_coords, tuple) and len(shape_or_coords) == 2 and isinstance(shape_or_coords[0], int):
        # input shape
        h, w = shape_or_coords
        return int(h * s), int(w * s)
    else:
        y_out, x_out = shape_or_coords
        if inverse:
            return x_out / s, y_out / s

# Rotação
def rotate_transform(shape_or_coords, dg=45, inverse=False):
    dg_rad = np.deg2rad(dg)
    cos_theta = np.cos(dg_rad)
    sin_theta = np.sin(dg_rad)
    if isinstance(shape_or_coords[0], int):
        h, w = shape_or_coords
        return h, w
    else:
        y_out, x_out = shape_or_coords
        cx, cy = IMG_W / 2, IMG_H / 2
        x_shift = x_out - cx
        y_shift = y_out - cy
        if inverse:
            x_in = cos_theta * x_shift + sin_theta * y_shift + cx
            y_in = -sin_theta * x_shift + cos_theta * y_shift + cy
            return x_in, y_in

# Cisalhamento
def shear_transform(shape_or_coords, shx=0.3, shy=0.2, inverse=False):
    if isinstance(shape_or_coords[0], int):
        h, w = shape_or_coords
        new_w = int(w + abs(shx) * h)
        new_h = int(h + abs(shy) * w)
        return new_h, new_w
    else:
        y_out, x_out = shape_or_coords
        if inverse:
            x_in = x_out - shy * y_out
            y_in = y_out - shx * x_out
            return x_in, y_in

# Cria pasta de saída
os.makedirs("resultados", exist_ok=True)

# Lista de transformações
transformacoes = [
    ("scale", scale_transform),
    ("rotate", rotate_transform),
    ("shear", shear_transform),
]

# Aplica transformações
for nome, func in transformacoes:
    img_nearest = apply_transform(img, func, interpolator="nearest")
    img_bicubic = apply_transform(img, func, interpolator="bicubic")
    cv2.imwrite(f"resultados/{nome}_nearest.png", img_nearest)
    cv2.imwrite(f"resultados/{nome}_bicubic.png", img_bicubic)

print("Imagens salvas na pasta `resultados/`.")
