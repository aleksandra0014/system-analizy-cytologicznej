import math
import numpy as np
from ultralytics import YOLO
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Lista modeli YOLO do przetestowania
models_paths = [
    r'runs\yolo_cells_detector_hybrid2\weights\best.pt',
    r'runs\yolo_cells_detector_hybrid3\weights\best.pt',
    r'runs\yolo_cells_detector_mine\weights\best.pt',
    r'runs\yolo_cells_detector2\weights\best.pt',
    r'runs\yolo_cells_detector3\weights\best.pt'
]

# Lista folderów testowych
test_folders = [
    r"data\LBC_slides\HSIL\pow 10",
    r"data\LBC_slides\LSIL\pow. 10",
    r"data\LBC_slides\NSIL\pow. 10"
]

# Liczba zdjęć do przetestowania z każdego folderu
num_samples = 15

# Katalog, gdzie będą zapisywane wyniki
output_root = r"C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\tests"

# Wybór wspólnych próbek dla każdego folderu (raz losowane)
test_samples_per_folder = {}

for folder in test_folders:
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    test_samples_per_folder[folder] = sample_files

for model_path in models_paths:
    print(f"\n>>> Testowanie modelu: {model_path}")
    model = YOLO(model_path)

    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(os.path.dirname(model_dir))

    for folder in test_folders:
        print(f"\nFolder testowy: {folder}")
        sample_files = test_samples_per_folder[folder]

        n_cols = 5
        n_rows = math.ceil(len(sample_files) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, file_name in enumerate(sample_files):
            img_path = os.path.join(folder, file_name)
            print(f"  Przetwarzanie: {img_path}")
            results = model(img_path, conf=0.2)
            result_img = results[0].plot()

            axes[idx].imshow(result_img)
            axes[idx].set_title(file_name, fontsize=10)
            axes[idx].axis("off")

        for j in range(len(sample_files), len(axes)):
            axes[j].axis("off")

        folder_parts = os.path.normpath(folder).split(os.sep)
        last_two = "_".join(folder_parts[-2:]).replace(".", "").replace(" ", "_")
        save_dir = os.path.join(output_root, model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{last_two}.png")

        plt.suptitle(f"Model: {model_name}\nFolder: {last_two}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Zapisano do: {save_path}")
