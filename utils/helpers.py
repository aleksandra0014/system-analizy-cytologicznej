import os
import random
import re
import shutil
import cv2
from matplotlib.path import Path
import pandas as pd
from typing import Dict, List, Tuple
from pandas import DataFrame
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from utils.create_syntetic_slides import segment_and_crop_cell

def get_folder_summary(path: str) -> DataFrame:
    summary = []

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            dimensions = []

            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.bmp'):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        try:
                            with Image.open(file_path) as img:
                                width, height = img.size
                                dimensions.append((width, height))
                        except Exception as e:
                            print(f"Błąd przy pliku {file_path}: {e}")

            if dimensions:
                widths, heights = zip(*dimensions)
                summary.append({
                    'Folder': folder_name,
                    'Liczba plików': len(dimensions),
                    'Minimalna szerokość': min(widths),
                    'Minimalna wysokość': min(heights),
                    'Maksymalna szerokość': max(widths),
                    'Maksymalna wysokość': max(heights)
                })
            else:
                summary.append({
                    'Folder': folder_name,
                    'Liczba plików': 0,
                    'Minimalna szerokość': None,
                    'Minimalna wysokość': None,
                    'Maksymalna szerokość': None,
                    'Maksymalna wysokość': None
                })

    return pd.DataFrame(summary)


def show_3_images_per_folder(base_path: str):
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    n_folders = len(folders)

    fig, axs = plt.subplots(n_folders, 3, figsize=(15, 5 * n_folders))

    for row_idx, folder_name in enumerate(folders):
        folder_path = os.path.join(base_path, folder_name)
        images = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.bmp'))]
        col_idx_list = [0, 10, 20]
        for col_idx in range(len(col_idx_list)):
            ax = axs[row_idx][col_idx]
            file_path = os.path.join(folder_path, images[col_idx_list[col_idx]])
            img = mpimg.imread(file_path)
            ax.imshow(img)
            ax.set_title(f"{folder_name}\n{images[col_idx_list[col_idx]]}", fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def check_resizing(img_path, size):
    new_width = size
    new_height = size
    target_size = (new_width, new_height)
    img = mpimg.imread(img_path)
    img_resized_cv2 = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    print(f"Wymiary obrazu po przeskalowaniu (OpenCV): {img_resized_cv2.shape}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Oryginalny Obraz\n{img.shape}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_resized_cv2)
    plt.title(f"Przeskalowany Obraz (OpenCV)\n{img_resized_cv2.shape}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()



def safe_filename(text):
    return re.sub(r'[^\w\-_\.]', '_', text)

def connect_folders(folders: Dict, output_folder: str, num_images: int, change_name: bool = True):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    os.makedirs(output_folder, exist_ok=True)

    used_filenames = set()

    for label, folder_path in folders.items():
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        selected_files = random.sample(files, min(num_images, len(files)))

        for idx, filename in enumerate(selected_files):
            input_path = os.path.join(folder_path, filename)

            try:
                img = Image.open(input_path).convert("RGB")  

                if change_name:
                    base_name = Path(filename).stem
                    base_name = safe_filename(base_name)
                    safe_label = safe_filename(label)
                    output_name = f"{safe_label}_{base_name}_{idx}.jpg"

                    while output_name in used_filenames:
                        rand_suffix = random.randint(1000, 9999)
                        output_name = f"{safe_label}_{base_name}_{idx}_{rand_suffix}.jpg"
                else:
                    output_name = safe_filename(f"{filename}_mendeley.jpg")

                used_filenames.add(output_name)
                output_path = os.path.join(output_folder, output_name)
                img.save(output_path)

            except Exception as e:
                print(f"Błąd przetwarzania pliku '{filename}': {e}")

        print(f"Gotowe! Z klasy '{label}' zapisano {len(selected_files)} obrazów.")

    print(f"\n✅ Całkowicie zapisano {len(used_filenames)} unikalnych obrazów do: {output_folder}")


def move_files_to_val(base_dir, proporcion):
    # === Ścieżki
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    val_dir = os.path.join(base_dir, "..", "val")  # np. ../val/
    val_images = os.path.join(val_dir, "images")
    val_labels = os.path.join(val_dir, "labels")

    # === Utwórz foldery val/
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)

    # === Pobierz listę obrazów
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)

    # === Wydziel 10%
    val_split = int(proporcion * len(image_files))
    val_subset = image_files[:val_split]

    # === Przenoszenie
    for file in val_subset:
        name = Path(file).stem
        shutil.move(os.path.join(images_dir, file), os.path.join(val_images, file))

        label_file = f"{name}.txt"
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_labels, label_file))

    print(f" Przeniesiono {len(val_subset)} plików do walidacji: {val_dir}")

def get_cropped_single_data(input_root, output_root):
    for class_name in os.listdir(input_root):
        input_class_dir = os.path.join(input_root, class_name)
        output_class_dir = os.path.join(output_root, class_name)

        if not os.path.isdir(input_class_dir):
            continue

        os.makedirs(output_class_dir, exist_ok=True)

        for file_name in os.listdir(input_class_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            input_path = os.path.join(input_class_dir, file_name)
            output_path = os.path.join(output_class_dir, file_name)

            img = cv2.imread(input_path)
            if img is None:
                print(f" Nie można wczytać: {input_path}")
                continue

            cropped_img, (w, h) = segment_and_crop_cell(img)
            resized_img = cv2.resize(cropped_img, (224, 224))
            cv2.imwrite(output_path, resized_img)
            print(f" Zapisano: {output_path}")

if __name__=='__main__':
    get_cropped_single_data(r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped')