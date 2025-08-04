import warnings
from ultralytics import YOLO
warnings.filterwarnings("ignore")

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import streamlit as st
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segmentation.models import UNet
from classification.preprocessing import apply_clahe

# === KONFIGURACJA ===
unet_model_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\unet_cell_nucleus_0208.pth"
yolo_model_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\models\yolo_detector_2107_100_20_16_7682\weights\best.pt'

image_size = (256, 256)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    tensor = transform(image)
    return image, tensor.unsqueeze(0)  # [1, C, H, W]

def predict_masks(model, image_tensor, device, threshold_nuclei=0.3, threshold_cell=0.5):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        preds = torch.sigmoid(output).squeeze(0)  # [2, H, W]

        mask_cell = (preds[0] > threshold_cell).float().cpu().numpy()
        mask_nucleus = (preds[1] > threshold_nuclei).float().cpu().numpy()

    return [mask_cell, mask_nucleus]



# 🔽 NOWA FUNKCJA: wybiera najlepsze jądro
def select_best_nucleus(mask, image_shape):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    h, w = image_shape
    center = np.array([w // 2, h // 2])

    best_score = -np.inf
    best_contour = None

    for cnt in contours:
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse
        if MA == 0 or ma == 0:
            continue

        owalnosc = min(MA, ma) / max(MA, ma)
        nucleus_center = np.array([x, y])
        dist_to_center = np.linalg.norm(nucleus_center - center)

        score = owalnosc / (dist_to_center + 1e-5)
        if score > best_score:
            best_score = score
            best_contour = cnt

    output_mask = np.zeros_like(mask, dtype=np.uint8)
    if best_contour is not None:
        cv2.drawContours(output_mask, [best_contour], -1, 255, thickness=-1)

    return output_mask

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
        areas = [(boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]) for k in overlaps]
        largest_idx = overlaps[np.argmax(areas)]
        selected_indices.append(largest_idx)
        suppressed[i] = True

    return selected_indices

def main():
    st.title("Analiza slajdu cytologicznego – UNet (maska jądra i komórki)")

    uploaded_file = st.file_uploader("Wgraj obraz slajdu (bmp, png, jpg)", type=["bmp", "png", "jpg"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_np = apply_clahe(image_np, clip_limit=2.0, tile_grid_size=(8, 8), use_median=True, median_kernel=3)

        # === MODELE ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet = UNet(in_channels=3, out_channels=2)
        unet.load_state_dict(torch.load(unet_model_path, map_location=device))
        unet.to(device)

        yolo = YOLO(yolo_model_path)

        # === DETEKCJA ===
        results = yolo(image_path, conf=0.25, iou=0.5, agnostic_nms=False)
        boxes_tensor = results[0].boxes.xyxy.cpu()
        classes_tensor = results[0].boxes.cls.cpu().int()

        boxes_array = boxes_tensor.numpy()
        keep_ids = nms_keep_largest_box(boxes_array, iou_thresh=0.4)

        boxes_tensor = boxes_tensor[keep_ids]
        classes_tensor = classes_tensor[keep_ids]

        cell_mask = (classes_tensor == 0) | (classes_tensor == 1)
        boxes = boxes_tensor[cell_mask].numpy().astype(int)

        crops, nuclei_masks, cell_masks = [], [], []

        for (x1, y1, x2, y2) in boxes:
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_crop:
                cv2.imwrite(tmp_crop.name, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                tmp_crop_path = tmp_crop.name

            _, image_tensor = preprocess_image(tmp_crop_path)
            masks = predict_masks(unet, image_tensor, device)

            mask_nucleus = cv2.resize(masks[1], (crop.shape[1], crop.shape[0]))
            best_nucleus = select_best_nucleus(mask_nucleus, image_shape=crop.shape[:2])

            mask_cell = cv2.resize(masks[0], (crop.shape[1], crop.shape[0])) * 255
 

            nuclei_masks.append(best_nucleus.astype(np.uint8))
            cell_masks.append(mask_cell.astype(np.uint8))
            crops.append(crop)

            os.remove(tmp_crop_path)

        # === SLAJD Z BBOX ===
        canvas = image_np.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.image(canvas, caption="Slajd z wykrytymi komórkami")

        # === SIATKA KOMÓREK ===
        st.subheader("Wycięte komórki z najlepszym jądrem i maską komórki")

        cols = 3
        for i in range(len(crops)):
            row = st.columns(cols)
            with row[0]:
                st.image(cv2.resize(crops[i], (128, 128)), caption=f"Komórka {i+1}")
            with row[1]:
                st.image(cv2.resize(nuclei_masks[i], (128, 128)), caption="Najlepsze jądro", channels="GRAY")
            with row[2]:
                st.image(cv2.resize(cell_masks[i], (128, 128)), caption="Maska komórki", channels="GRAY")

        os.remove(image_path)

if __name__ == "__main__":
    main()
