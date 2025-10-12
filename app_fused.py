import warnings
from ultralytics import YOLO
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import streamlit as st
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segmentation.modelsUnet import UNet
from segmentation.features import extract_features
from classification.models import CytologyClassifier
import joblib
import pandas as pd


# === KONFIGURACJA ===
unet_model_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\unet_cell_nucleus_0208.pth"
yolo_model_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\models\yolo_detector_2107_100_20_16_7682\weights\best.pt'
xgb_model_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\best_model_XGBoost.pkl"
label_encoder_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\label_encoder.pkl'
scaler_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\scaler.pkl'

# VGG
CLASS_NAMES = ['HSIL', 'LSIL', 'NSIL']
vgg_weights = r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\vgg16\32_0_0001_50_0608.pth'
ARCHITECTURE = 'vgg16'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === MODELE ===
unet = UNet(in_channels=3, out_channels=2)
unet.load_state_dict(torch.load(unet_model_path, map_location=device))
unet.to(device).eval()

yolo = YOLO(yolo_model_path)

xgb_model = joblib.load(xgb_model_path)
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)

vgg_clf = CytologyClassifier(num_classes=len(CLASS_NAMES), architecture=ARCHITECTURE)
vgg_clf.load(vgg_weights)
vgg_clf.model.eval().to(device)


# === FUNKCJE ===
def preprocess_image_for_vgg(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return tfm(img)

@torch.inference_mode()
def predict_vgg_probs(model_or_wrapper, crop: np.ndarray) -> np.ndarray:
    img_pil = Image.fromarray(crop).convert("RGB")
    x = preprocess_image_for_vgg(img_pil).unsqueeze(0).to(device)
    net = getattr(model_or_wrapper, "model", model_or_wrapper)
    logits = net(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs

def predict_masks(model, image_tensor, device, threshold_nuclei=0.3, threshold_cell=0.5):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        preds = torch.sigmoid(output).squeeze(0)  # [2, H, W]

        mask_cell = (preds[0] > threshold_cell).float().cpu().numpy()
        mask_nucleus = (preds[1] > threshold_nuclei).float().cpu().numpy()

    return [mask_cell, mask_nucleus]

def preprocess_image_unet(image_path, size=256):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return image, transform(image).unsqueeze(0)

def select_best_nucleus(mask, image_shape):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    h, w = image_shape
    center = np.array([w // 2, h // 2])
    best_score, best_contour = -np.inf, None
    for cnt in contours:
        if len(cnt) < 5: continue
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), _ = ellipse
        if MA == 0 or ma == 0: continue
        owalnosc = min(MA, ma) / max(MA, ma)
        dist_to_center = np.linalg.norm(np.array([x, y]) - center)
        score = owalnosc / (dist_to_center + 1e-5)
        if score > best_score:
            best_score, best_contour = score, cnt
    out = np.zeros_like(mask, dtype=np.uint8)
    if best_contour is not None:
        cv2.drawContours(out, [best_contour], -1, 255, -1)
    return out

def predict_xgb_probs(features: dict) -> np.ndarray:
    FEATURE_NAMES = ['N', 'C', 'NCr', 'Np', 'Cp', 'NCp', 'MinA', 'MinAr', 'MaxA', 'MaxAr',
                     'Nar', 'Car', 'NCar', 'NExt', 'CExt', 'NCExt', 'NSol', 'CSol', 'NCs',
                     'EqN', 'EqC', 'NCEq', 'OrN', 'OrC', 'NCOr']
    X_new = pd.DataFrame([[features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(X_new)
    probs = xgb_model.predict_proba(X_scaled)[0]
    probs_aligned = np.zeros(len(CLASS_NAMES))
    for i, cname in enumerate(label_encoder.classes_):
        probs_aligned[CLASS_NAMES.index(cname)] = probs[i]
    return probs_aligned / probs_aligned.sum()

def fuse_probs(p1, p2, eps=1e-9):
    p = (p1+eps) * (p2+eps)
    return p / p.sum()


# === STREAMLIT ===
def main():
    st.title("Analiza cytologiczna – VGG + XGBoost + fuzja")

    uploaded_file = st.file_uploader("Wgraj obraz slajdu", type=["bmp", "png", "jpg"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        results = yolo(image_path, conf=0.25, iou=0.5)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        crops, nuclei_masks, cell_masks = [], [], []
        for (x1, y1, x2, y2) in boxes:
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0: continue

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_crop:
                cv2.imwrite(tmp_crop.name, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                tmp_path = tmp_crop.name

            _, tensor = preprocess_image_unet(tmp_path)
            masks = predict_masks(unet, tensor, device)
            mask_nucleus = cv2.resize(masks[1], (crop.shape[1], crop.shape[0]))
            best_nucleus = select_best_nucleus(mask_nucleus, crop.shape[:2])
            mask_cell = cv2.resize(masks[0], (crop.shape[1], crop.shape[0])) * 255

            features = extract_features(best_nucleus, mask_cell)

            probs_vgg = predict_vgg_probs(vgg_clf, crop)
            probs_xgb = predict_xgb_probs(features)
            probs_fused = fuse_probs(probs_vgg, probs_xgb)

            crops.append((crop, probs_vgg, probs_xgb, probs_fused))

            os.remove(tmp_path)

        for i, (crop, pv, px, pf) in enumerate(crops):
            st.image(crop, caption=f"Komórka {i+1}", channels="RGB")
            st.write("**VGG probs**:", dict(zip(CLASS_NAMES, np.round(pv, 3))))
            st.write("**XGB probs**:", dict(zip(CLASS_NAMES, np.round(px, 3))))
            st.write("**Fused probs**:", dict(zip(CLASS_NAMES, np.round(pf, 3))))
            st.write(f"**Pred class:** {CLASS_NAMES[np.argmax(pf)]}")

        os.remove(image_path)


if __name__ == "__main__":
    main()
