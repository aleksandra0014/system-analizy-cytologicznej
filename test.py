import os
import cv2
import numpy as np

# Twoja funkcja CLAHE
def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
        use_median: bool = False,
        median_kernel: int = 3
    ) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if use_median:
            l = cv2.medianBlur(l, median_kernel)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return result
    else:
        raise ValueError("Input image must be a 3-channel (RGB/BGR) image.")

# === Ścieżki ===
input_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\single_all"
output_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\single_all_clahe"
os.makedirs(output_dir, exist_ok=True)

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# === Przetwarzanie ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(valid_exts):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    image = cv2.imread(input_path)
    if image is None:
        print(f"⚠️ Nie udało się wczytać: {filename}")
        continue

    try:
        image_clahe = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), use_median=True)
        cv2.imwrite(output_path, image_clahe)
        print(f"✅ Zapisano: {output_path}")
    except Exception as e:
        print(f"❌ Błąd przy {filename}: {e}")
