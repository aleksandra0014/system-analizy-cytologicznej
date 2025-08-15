import os
import sys

import cv2
import numpy as np

import random
import shutil
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.create_syntetic_slides import segment_and_crop_cell


def get_cropped_and_split_data(input_root: str, output_root: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> None:
    """
    Crops and resizes images from input_root/class_name folders,
    then randomly splits them into train/val/test folders while preserving class structure.

    Args:
        input_root (str): Path to root directory with class folders.
        output_root (str): Path to save split and processed data.
        train_ratio (float): Proportion of data used for training.
        val_ratio (float): Proportion of data used for validation.
        test_ratio (float): Proportion of data used for testing.

    Returns:
        None
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    for class_name in os.listdir(input_root):
        input_class_dir = os.path.join(input_root, class_name)

        if not os.path.isdir(input_class_dir):
            continue

        image_files = [f for f in os.listdir(input_class_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        random.shuffle(image_files)

        total = len(image_files)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        subsets = {
            "train": image_files[:n_train],
            "val": image_files[n_train:n_train + n_val],
            "test": image_files[n_train + n_val:]
        }

        for subset, files in subsets.items():
            output_class_dir = os.path.join(output_root, subset, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            for file_name in files:
                input_path = os.path.join(input_class_dir, file_name)
                output_path = os.path.join(output_class_dir, file_name)

                img = cv2.imread(input_path)
                if img is None:
                    print(f"Nie można wczytać: {input_path}")
                    continue

                cropped_img, _ = segment_and_crop_cell(img)
                resized_img = cv2.resize(cropped_img, (224, 224))
                cv2.imwrite(output_path, resized_img)
                print(f"Zapisano: {output_path}")


def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
        use_median: bool = False,
        median_kernel: int = 3
    ) -> np.ndarray:
        """
        Apply CLAHE to a RGB image.

        Args:
            image (np.ndarray): Input image.
            clip_limit (float): Maximum histogram height (Hmax).
            tile_grid_size (tuple): Block size (e.g., (8,8)).
            use_median (bool): Whether to apply median filter before CLAHE.
            median_kernel (int): Median window size (must be odd).

        Returns:
            np.ndarray: CLAHE-processed image (same shape as input).
        """
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

if __name__=='__main__':
    get_cropped_and_split_data(r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped3')