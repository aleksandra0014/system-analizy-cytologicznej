import sys
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

from classification.models import CytologyClassifier
from xai_test import plot_gradcam_results2

from classification.preprocessing import apply_clahe

# === CONFIG ===
yolo_model_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\models\yolo_detector_2107_100_20_16_7682\weights\best.pt'
classifier_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\resnet18\32_0_001_50_0308.pth'
architecture = 'resnet18'  # 'resnet18' or 'custom_cnn'
class_names = ['HSIL', 'LSIL', 'NSIL']


classifier_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])  # old version for RGB input

def predict_label(classifier, crop_np):
    image_tensor = classifier_transform(crop_np).unsqueeze(0).to(classifier.device)
    with torch.no_grad():
        output = classifier.model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax().item()
        return classifier.class_names[pred_idx], image_tensor

def main():
    st.title("Analiza slajdu cytologicznego")
    uploaded_file = st.file_uploader("Wgraj obraz slajdu (bmp, png, jpg)", type=["bmp", "png", "jpg"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_np = apply_clahe(image_np, clip_limit=2.0, tile_grid_size=(8, 8), use_median=True, median_kernel=3)

        # === MODELE ===
        yolo = YOLO(yolo_model_path)
        classifier = CytologyClassifier(num_classes=3, architecture=architecture)
        classifier.load(classifier_path)

        # === DETEKCJA ===
        results = yolo(image_path)
        boxes_tensor = results[0].boxes.xyxy.cpu()
        classes_tensor = results[0].boxes.cls.cpu().int()

        # Filtruj tylko obiekty oznaczone jako "cell" (klasa 0) lub "HSIL" (klasa 1)
        cell_mask = (classes_tensor == 0) | (classes_tensor == 1)  # 0: cell, 1: HSIL
        boxes = boxes_tensor[cell_mask].numpy().astype(int)
        yolo_labels = classes_tensor[cell_mask].numpy().astype(int)

        crops, labels, tensors = [], [], []

        for (x1, y1, x2, y2), yolo_label in zip(boxes, yolo_labels):
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            if yolo_label == 1:
                label = "HSIL/LSIL group"
                tensor = None
            else:
                label, tensor = predict_label(classifier, crop)
            crops.append(crop)
            labels.append(label)
            tensors.append(tensor)

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

        # Sort cells by label: HSIL, LSIL, NSIL, then others
        label_order = {"HSIL": 0, "LSIL": 1, "NSIL": 2}
        sorted_data = sorted(zip(crops, labels, tensors), key=lambda x: label_order.get(x[1], 99))
        crops, labels, tensors = zip(*sorted_data) if sorted_data else ([], [], [])

        # === SIATKA KOMÓREK ===
        st.subheader("Wycięte komórki z klasyfikacjami")
        cols = 4
        rows = (len(crops) + cols - 1) // cols
        for r in range(rows):
            row = st.columns(cols)
            for c in range(cols):
                idx = r * cols + c
                if idx < len(crops):
                    with row[c]:
                        img = cv2.resize(crops[idx], (128, 128))
                        st.image(img, caption=f"{idx + 1}. {labels[idx]}")


        # === GRAD-CAM ===
        # st.subheader("GradCAM dla każdej komórki")
        # for i, (tensor, label) in enumerate(zip(tensors, labels)):
        #     if label == "HSIL/LSIL group":
        #         continue
        #     st.markdown(f"**Komórka {i+1} ({label})**")
        #     fig = plot_gradcam_results2(classifier.model, tensor, class_names=class_names, transform=None, model_name=architecture)
        #     st.pyplot(fig)

        os.remove(image_path)

if __name__ == "__main__":
    main()
