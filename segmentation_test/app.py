import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import streamlit as st
import tempfile
import os

from segments_func import *

# === CONFIG ===
yolo_model_path = r'C:/Users/aleks/OneDrive/Documents/inzynierka/runs/yolo_cells_detector_hybrid3/weights/best.pt'
class_names = ['HSIL', 'LSIL', 'NSIL']
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xa1, ya1, xa2, ya2 = box2

    xi1, yi1 = max(x1, xa1), max(y1, ya1)
    xi2, yi2 = min(x2, xa2), min(y2, ya2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xa2 - xa1) * (ya2 - ya1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def nms_keep_largest_box(boxes, iou_thresh=0.4):
    boxes = np.array(boxes)
    num_boxes = len(boxes)
    suppressed = np.zeros(num_boxes, dtype=bool)

    selected_indices = []

    for i in range(num_boxes):
        if suppressed[i]:
            continue
        overlaps = [i]
        for j in range(num_boxes):
            if i != j and not suppressed[j]:
                iou = compute_iou(boxes[i], boxes[j])
                if iou >= iou_thresh:
                    overlaps.append(j)
                    suppressed[j] = True
        # wybierz największy box z tej grupy overlapów
        areas = [(boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]) for k in overlaps]
        largest_idx = overlaps[np.argmax(areas)]
        selected_indices.append(largest_idx)
        suppressed[i] = True

    return selected_indices


def predict(image): 
    get_mean_ratio_file(image)

def classify_by_ratio(ratio):
    if NSIL_range[0] <= ratio <= NSIL_range[1]:
        return "NSIL"
    elif LSIL_range[0] <= ratio <= LSIL_range[1]:
        return "LSIL"
    elif HSIL_range[0] <= ratio <= HSIL_range[1]:
        return "HSIL"
    else:
        return "Nieznana"

def main():
    st.title("Analiza slajdu cytologicznego")
    uploaded_file = st.file_uploader("Wgraj obraz slajdu (bmp, png, jpg)", type=["bmp", "png", "jpg"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # === MODELE ===
        yolo = YOLO(yolo_model_path)

        # === DETEKCJA ===
        results = yolo(image_path, conf=0.25, iou=0.5, agnostic_nms=False)

        boxes_tensor = results[0].boxes.xyxy.cpu()
        classes_tensor = results[0].boxes.cls.cpu().int()

        # Zastosuj własne NMS bazujące na rozmiarze (zamiast klasycznego confidence-based)
        boxes_array = boxes_tensor.numpy()
        keep_ids = nms_keep_largest_box(boxes_array, iou_thresh=0.4)

        boxes_tensor = boxes_tensor[keep_ids]
        classes_tensor = classes_tensor[keep_ids]

        # Filtruj tylko obiekty oznaczone jako "cell" (klasa 0) lub "HSIL" (klasa 1)
        cell_mask = (classes_tensor == 0) | (classes_tensor == 1)
        boxes = boxes_tensor[cell_mask].numpy().astype(int)
        yolo_labels = classes_tensor[cell_mask].numpy().astype(int)


        crops, labels, tensors, ratios, nuclei_masks = [], [], [], [], []

        for (x1, y1, x2, y2), yolo_label in zip(boxes, yolo_labels):
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            if yolo_label == 1:
                label = "HSIL/LSIL group"
                tensor = None
                nuclei_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            else:
                # Zapisz crop do pliku tymczasowego i licz ratio oraz maskę jądra
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_crop:
                    cv2.imwrite(tmp_crop.name, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    tmp_crop_path = tmp_crop.name
                # Pobierz maskę jądra i ratio
                result = extract_nucleus_and_cell_contours(tmp_crop_path)
                if result is not None:
                    _, mask_nucleus, _, _, _ = result
                else:
                    mask_nucleus = np.zeros(crop.shape[:2], dtype=np.uint8)
                ratio = get_mean_ratio_file(tmp_crop_path)
                ratios.append(ratio)
                os.remove(tmp_crop_path)
                label = classify_by_ratio(ratio)
                tensor = None
                nuclei_mask = mask_nucleus
            crops.append(crop)
            labels.append(label)
            tensors.append(tensor)
            nuclei_masks.append(nuclei_mask)

        # === SLAJD Z BBOX ===
        canvas = image_np.copy()
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        st.image(canvas, caption="Slajd z wykrytymi komórkami")

        # === PODSUMOWANIE ===
        st.subheader("Podsumowanie klas")
        for cls in class_names:
            st.write(f"{cls}: {labels.count(cls)} komórek")
        st.write(f"HSIL/LSIL group: {labels.count('HSIL/LSIL group')} komórek")

        # === SIATKA KOMÓREK ===
        st.subheader("Wycięte komórki z klasyfikacjami i maskami jąder")
        cols = 4
        display_indices = []
        ratios_display = []
        nuclei_display = []
        ratio_idx = 0
        for i, label in enumerate(labels):
            if label != "HSIL/LSIL group":
                display_indices.append(i)
                ratios_display.append(ratios[ratio_idx])
                nuclei_display.append(nuclei_masks[i])
                ratio_idx += 1
        rows = (len(display_indices) + cols - 1) // cols
        for r in range(rows):
            row = st.columns(cols)
            for c in range(cols):
                idx_in_display = r * cols + c
                if idx_in_display < len(display_indices):
                    idx = display_indices[idx_in_display]
                    img = cv2.resize(crops[idx], (128, 128))
                    mask = cv2.resize(nuclei_display[idx_in_display], (128, 128))
                    # Dwie kolumny obok siebie: komórka i maska
                    with row[c]:
                        st.image(img, caption=f"{idx + 1}. {labels[idx]}, ratio: {ratios_display[idx_in_display]:.2f}%")
                        st.image(mask, caption="Maska jądra", channels="GRAY")
        os.remove(image_path)

if __name__ == "__main__":
    main()
