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
from create_syntetic_slides import segment_and_crop_cell
import numpy as np
import matplotlib.pyplot as plt
import os

def get_folder_summary(path: str) -> DataFrame:
    """
    Summarizes image dimensions for each folder in the given path.
    Args:
        path (str): Path to the root directory containing folders of images.
    Returns:
        DataFrame: Summary statistics for each folder (file count, min/max width/height).
    """
    summary = []

    def process_folder(folder_path, folder_name_prefix=""):
        dimensions = []
        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)
            if os.path.isdir(entry_path):
                # Rekurencyjnie przetwarzaj podfoldery
                process_folder(entry_path, folder_name_prefix + entry + "/")
            elif entry.lower().endswith((".bmp", ".jpg", ".png")) and os.path.isfile(entry_path):
                try:
                    with Image.open(entry_path) as img:
                        width, height = img.size
                        dimensions.append((width, height))
                except Exception as e:
                    print(f"Błąd przy pliku {entry_path}: {e}")
        # Dodaj podsumowanie tylko jeśli folder zawiera obrazy lub jest pusty (nie liczymy podfolderów)
        if dimensions or not any(os.path.isdir(os.path.join(folder_path, f)) for f in os.listdir(folder_path)):
            if dimensions:
                widths, heights = zip(*dimensions)
                summary.append({
                    'Folder': folder_name_prefix.rstrip("/"),
                    'Liczba plików': len(dimensions),
                    'Minimalna szerokość': min(widths),
                    'Minimalna wysokość': min(heights),
                    'Maksymalna szerokość': max(widths),
                    'Maksymalna wysokość': max(heights)
                })
            else:
                summary.append({
                    'Folder': folder_name_prefix.rstrip("/"),
                    'Liczba plików': 0,
                    'Minimalna szerokość': None,
                    'Minimalna wysokość': None,
                    'Maksymalna szerokość': None,
                    'Maksymalna wysokość': None
                })

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            process_folder(folder_path, folder_name_prefix=folder_name)

    return pd.DataFrame(summary)


def show_3_images_per_folder(base_path: str):
    """
    Displays 3 sample images from each folder in the base_path using matplotlib.
    Args:
        base_path (str): Path to the root directory containing folders of images.
    """
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


def check_resizing(img_path: str, size: int) -> None:
    """
    Loads an image, resizes it to the given size using OpenCV, and displays original and resized images.
    Args:
        img_path (str): Path to the image file.
        size (int): Target size for width and height.
    Returns:
        None
    """
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


def safe_filename(text: str) -> str:
    """
    Converts a string to a safe filename by replacing unsafe characters with underscores.
    Args:
        text (str): Input string.
    Returns:
        str: Safe filename string.
    """
    return re.sub(r'[^\w\-_\.]', '_', text)

def connect_folders(
    folders: Dict[str, str],
    output_folder: str,
    num_images: int,
    change_name: bool = True
) -> None:
    """
    Copies a specified number of images from each folder to an output folder, optionally renaming them.
    Args:
        folders (Dict[str, str]): Mapping of class labels to folder paths.
        output_folder (str): Path to the output directory.
        num_images (int): Number of images to copy per class.
        change_name (bool): Whether to rename images for uniqueness.
    Returns:
        None
    """
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


def move_files_to_val(base_dir: str, proporcion: float) -> None:
    """
    Moves a proportion of images and their labels from the training folder to a validation folder.
    Args:
        base_dir (str): Path to the base directory containing 'images' and 'labels' folders.
        proporcion (float): Proportion of files to move to validation (0 < proporcion < 1).
    Returns:
        None
    """
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    val_dir = os.path.join(base_dir, "..", "val")  # np. ../val/
    val_images = os.path.join(val_dir, "images")
    val_labels = os.path.join(val_dir, "labels")

    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)

    val_split = int(proporcion * len(image_files))
    val_subset = image_files[:val_split]

    for file in val_subset:
        name = Path(file).stem
        shutil.move(os.path.join(images_dir, file), os.path.join(val_images, file))

        label_file = f"{name}.txt"
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_labels, label_file))

    print(f" Przeniesiono {len(val_subset)} plików do walidacji: {val_dir}")


