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
import joblib

from classification.models import CytologyClassifier, SVMClassifier
from xai_test import plot_gradcam_results2

# === CONFIG ===
yolo_model_path = r'C:/Users/aleks/OneDrive/Documents/inzynierka/runs/yolo_cells_detector_hybrid3/weights/best.pt'
classifier_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification_models\vgg16\24_0.001_50_07.07.pth'
svm_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\svm_vgg16.pkl'  # Update path if needed
architecture = 'vgg16'  # 'resnet18' or 'custom_cnn'
class_names = ['HSIL', 'LSIL', 'NSIL']


classifier_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])  # old version for RGB input


def extract_resnet_features(model, crop_np, device):
    image_tensor = classifier_transform(crop_np).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.model.conv1(image_tensor)
        x = model.model.bn1(x)
        x = model.model.relu(x)
        x = model.model.maxpool(x)
        x = model.model.layer1(x)
        x = model.model.layer2(x)
        x = model.model.layer3(x)
        x = model.model.layer4(x)
        x = model.model.avgpool(x)
        feats = torch.flatten(x, 1).cpu().numpy()
    return feats, image_tensor

def extract_vgg_features(model, crop_np, device):
    image_tensor = classifier_transform(crop_np).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.model.features(image_tensor)  
        x = x.view(x.size(0), -1)        
        feats = x.cpu().numpy()
    return feats, image_tensor

# SVM prediction for a single crop
def predict_label_svm(classifier, svm, crop_np):
    feats, image_tensor = extract_vgg_features(classifier, crop_np, classifier.device)
    pred_idx = svm.predict(feats)[0]
    return classifier.class_names[pred_idx], image_tensor

def main():
    st.title("Analiza slajdu cytologicznego (SVM)")
    uploaded_file = st.file_uploader("Wgraj obraz slajdu (bmp, png, jpg)", type=["bmp", "png", "jpg"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # === MODELE ===
        yolo = YOLO(yolo_model_path)
        classifier = CytologyClassifier(num_classes=3, architecture=architecture)
        classifier.load(classifier_path)
        svm = joblib.load(svm_path)

        # === DETEKCJA ===
        results = yolo(image_path)
        # Współrzędne i klasy detekcji
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
                label, tensor = predict_label_svm(classifier, svm, crop)
            crops.append(crop)
            labels.append(label)
            tensors.append(tensor)

        # === SLAJD Z BBOX ===
        canvas = image_np.copy()
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        st.image(canvas, caption="Slajd z wykrytymi komórkami (SVM)")

        # === PODSUMOWANIE ===
        st.subheader("Podsumowanie klas (SVM)")
        for cls in class_names:
            st.write(f"{cls}: {labels.count(cls)} komórek")
        st.write(f"HSIL/LSIL group: {labels.count('HSIL/LSIL group')} komórek")

        # === SIATKA KOMÓREK ===
        st.subheader("Wycięte komórki z klasyfikacjami (SVM)")
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
        st.subheader("GradCAM dla każdej komórki")
        for i, (tensor, label) in enumerate(zip(tensors, labels)):
            st.markdown(f"**Komórka {i+1} ({label})**")
            if tensor is not None:
                fig = plot_gradcam_results2(classifier.model, tensor, class_names=class_names, transform=None, model_name='vgg16')
                st.pyplot(fig)

        os.remove(image_path)

if __name__ == "__main__":
    main()
