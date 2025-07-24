import os
import cv2
import numpy as np
import random
from pathlib import Path
from PIL import Image
import shutil

def segment_and_crop_cell(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Segments and crops a cell from the input image using HSV color thresholding.
    Args:
        img (np.ndarray): Input image, possibly with alpha channel.
    Returns:
        tuple: Cropped cell image and its (width, height).
    """
    img_rgb = img[:, :, :3] if img.shape[2] == 4 else img
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 10, 10])
    upper = np.array([180, 255, 240])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        margin = 5
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, img.shape[1] - x)
        h = min(h + 2 * margin, img.shape[0] - y)
        cropped = img[y:y+h, x:x+w]
        return cropped, (w, h)
    return img, (img.shape[1], img.shape[0])


def resize_cell(
    cell: np.ndarray,
    original_w: int,
    original_h: int,
    min_size: int = 64,
    max_size: int = 128
) -> tuple[np.ndarray, int, int]:
    """
    Resizes the cell image with a random scale factor within the given size range.
    Args:
        cell (np.ndarray): Cropped cell image.
        original_w (int): Original width of the cell.
        original_h (int): Original height of the cell.
        min_size (int): Minimum size for scaling.
        max_size (int): Maximum size for scaling.
    Returns:
        tuple: Resized cell image, new width, new height.
    """
    scale = random.uniform(min_size / max(original_h, original_w), max_size / max(original_h, original_w))
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    resized = cv2.resize(cell, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, new_w, new_h

def overlaps_soft(
    x1: int,
    y1: int,
    w1: int,
    h1: int,
    boxes: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.1
) -> bool:
    """
    Checks for overlap between a proposed bounding box and a list of existing boxes using IoU threshold.
    Args:
        x1, y1 (int): Top-left coordinates of the proposed box.
        w1, h1 (int): Width and height of the proposed box.
        boxes (list): List of existing boxes as (x, y, w, h).
        iou_thresh (float): IoU threshold for overlap.
    Returns:
        bool: True if overlap exceeds threshold, False otherwise.
    """
    box1 = [x1, y1, x1+w1, y1+h1]
    for (x2, y2, w2, h2) in boxes:
        box2 = [x2, y2, x2+w2, y2+h2]
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        iou = inter_area / float(box1_area + box2_area - inter_area)
        if iou > iou_thresh:
            return True
    return False


if __name__ == "__main__":
    CELL_FOLDER = r'C:\Users\aleks\OneDrive\Documents\inzynierka\single_all'
    EXTRA_FOLDERS = [r'C:\Users\aleks\OneDrive\Documents\inzynierka\data_single\LSIL', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data_single\NSIL', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data_single\HSIL']

    OUTPUT_IMAGES = 'images'
    OUTPUT_LABELS = 'labels'
    IMAGE_SIZE = (1024, 1024)
    CELLS_PER_IMAGE = (5, 50)
    NUM_IMAGES = 200

    os.makedirs(CELL_FOLDER, exist_ok=True)
    for folder in EXTRA_FOLDERS:
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            for file in Path(folder).glob(ext):
                shutil.copy(file, CELL_FOLDER)

    bmp_files = list(Path(CELL_FOLDER).glob("*.bmp"))
    for bmp_path in bmp_files:
        img = Image.open(bmp_path).convert("RGB")
        new_name = bmp_path.with_suffix('.jpg')
        img.save(new_name)

    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_LABELS, exist_ok=True)
    
    cell_paths = list(Path(CELL_FOLDER).glob("*.png")) + \
                list(Path(CELL_FOLDER).glob("*.jpg")) + \
                list(Path(CELL_FOLDER).glob("*.jpeg"))
    cells = []
    for p in cell_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        cropped, (w, h) = segment_and_crop_cell(img)
        cells.append((cropped, w, h))

    for i in range(NUM_IMAGES):
        canvas = np.ones((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8) * 255
        label_lines = []
        placed_boxes = []

        num_cells = random.randint(*CELLS_PER_IMAGE)
        attempts = 0
        max_attempts = 100 * num_cells

        while len(placed_boxes) < num_cells and attempts < max_attempts:
            attempts += 1
            cell, orig_w, orig_h = random.choice(cells)
            cell, w, h = resize_cell(cell, orig_w, orig_h)

            max_x = IMAGE_SIZE[0] - w
            max_y = IMAGE_SIZE[1] - h
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            if overlaps_soft(x, y, w, h, placed_boxes):
                continue

            placed_boxes.append((x, y, w, h))

            if cell.shape[2] == 4:
                alpha = cell[:, :, 3] / 255.0
                for c in range(3):
                    canvas[y:y+h, x:x+w, c] = canvas[y:y+h, x:x+w, c] * (1 - alpha) + cell[:, :, c] * alpha
            else:
                canvas[y:y+h, x:x+w] = cell

            x_center = (x + w / 2) / IMAGE_SIZE[0]
            y_center = (y + h / 2) / IMAGE_SIZE[1]
            bbox_width = w / IMAGE_SIZE[0]
            bbox_height = h / IMAGE_SIZE[1]
            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

            x1 = int((x_center - bbox_width / 2) * IMAGE_SIZE[0])
            y1 = int((y_center - bbox_height / 2) * IMAGE_SIZE[1])
            x2 = int((x_center + bbox_width / 2) * IMAGE_SIZE[0])
            y2 = int((y_center + bbox_height / 2) * IMAGE_SIZE[1])

        image_name = f"synth_{i:04d}.jpg"
        label_name = f"synth_{i:04d}.txt"

        cv2.imwrite(os.path.join(OUTPUT_IMAGES, image_name), canvas)
        with open(os.path.join(OUTPUT_LABELS, label_name), 'w') as f:
            f.write("\n".join(label_lines))

    print("Gotowe! W folderach 'images/' i 'labels/' masz dane do trenowania YOLO.")


