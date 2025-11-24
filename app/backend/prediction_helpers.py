import datetime
import warnings
from ultralytics import YOLO

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

import cv2
from PIL import Image
from torchvision import transforms

import tempfile
import os
import joblib

from segmentation.modelsUnet import UNet4Levels, preprocess_image, predict_masks, UNet
from segmentation.features import extract_features
from classification.models import CytologyClassifier, predict_label

from app.backend.helpers import *  

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from pathlib import Path

# Ensure we load the .env located in the backend folder (app/backend/.env)
# using an absolute path so imports/run-from-root still pick it up.
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    # fallback to default behavior (looks in CWD and parents)
    load_dotenv()

CLASS_NAMES = os.getenv("CLASS_NAMES", "HSIL,LSIL,NSIL").split(",")
ARCHITECTURE = os.getenv("ARCHITECTURE", "resnet18")
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", r"C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\resnet18\16_0_0001_50_1110.pth")
UNET_MODEL_PATH = os.getenv("UNET_MODEL_PATH", r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\unet4_cell_nucleus_4_50_1310.pth")
THRESHOLD_NUCLEI = float(os.getenv("THRESHOLD_NUCLEI", 0.7))
THRESHOLD_CELLS = float(os.getenv("THRESHOLD_CELLS", 0.3))
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", r"C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\models\yolo_detector_2107_100_20_16_7682\weights\best.pt")
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\models_paths\best_model_RandomForest_311.pkl")
API_KEY = os.getenv("API_KEY", os.getenv("api_key", ''))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


unet = UNet(in_channels=3, out_channels=2)
unet.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
unet.to(device).eval()

yolo = YOLO(YOLO_MODEL_PATH)

ml_model = joblib.load(ML_MODEL_PATH)
label_encoder = ml_model['label_encoder']

# fused_model = joblib.load(FUSED_MODEL_PATH)
# label_encode_fused = fused_model['label_encoder']

cnn_classifier = CytologyClassifier(num_classes=len(CLASS_NAMES), architecture=ARCHITECTURE)
cnn_classifier.load(CNN_MODEL_PATH)
cnn_classifier.model.eval().to(device)

def fuse_func(p1, p2, eps=1e-9):
    p = (p1 + eps) + (p2 + eps)
    return p / p.sum()

def get_info(image_path, show_image=True):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    results = yolo(image_path, conf=0.25, iou=0.5, agnostic_nms=False)
    boxes_tensor = results[0].boxes.xyxy.cpu()
    classes_tensor = results[0].boxes.cls.cpu().int()

    boxes_array = boxes_tensor.numpy()
    keep_ids = nms_keep_largest_box(boxes_array, iou_thresh=0.4)

    boxes_tensor = boxes_tensor[keep_ids]
    classes_tensor = classes_tensor[keep_ids]

    cell_mask = (classes_tensor == 0) | (classes_tensor == 1) | (classes_tensor == 2)
    boxes = boxes_tensor[cell_mask].numpy().astype(int)

    predicted_classes_cnn = {}
    predicted_classed_ml = {}
    predicted_classed_fused = {}
    features_list = {}
    probs = {}
    crop_paths = {}
    probs_list = []

    bbox_image_path = None
    if show_image:
        fig, ax = plt.subplots(figsize=(7.68, 5.12), dpi=100)
        ax.imshow(image_np)
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor= '#f7c873',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, max(0, y1 - 5), str(idx),
                color='#3a5ba0',  
                fontsize=12, weight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )
        ax.set_axis_off()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_bbox:
            fig.savefig(tmp_bbox.name, bbox_inches='tight', pad_inches=0, dpi=100)
            bbox_image_path = tmp_bbox.name
        plt.close(fig)
        im = Image.open(bbox_image_path)
        im = im.resize((768, 512), Image.LANCZOS)
        im.save(bbox_image_path)

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        yolo_class = None
        for i, box in enumerate(boxes_tensor.numpy().astype(int)):
            if np.array_equal(box, [x1, y1, x2, y2]):
                yolo_class = int(classes_tensor[i].item())
                break

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_crop:
            cv2.imwrite(tmp_crop.name, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            tmp_path = tmp_crop.name
            crop_paths[idx] = tmp_path

        if yolo_class == 1:
            predicted_classes_cnn[idx] = 'HSIL/LSIL_group'
            predicted_classed_fused[idx] = 'HSIL/LSIL_group'
            probs_list.append([0.7, 0.3, 0])
            probs[idx] = {
            'fused': [0.7, 0.3, 0]
            }
            continue

        _, tensor = preprocess_image(tmp_path)
        masks = predict_masks(unet, tensor, device, threshold_nuclei=THRESHOLD_NUCLEI, threshold_cell=THRESHOLD_CELLS)
        mask_nucleus = cv2.resize(masks[1], (crop.shape[1], crop.shape[0]))
        best_nucleus = select_best_nucleus(mask_nucleus, crop.shape[:2])
        mask_cell = cv2.resize(masks[0], (crop.shape[1], crop.shape[0])) * 255

        features = extract_features(best_nucleus, mask_cell)
        
        if features is None or len(features) == 0:
            print(f"Pominięto komórkę {idx} - brak features")
            continue
            
        if all(value == 0 for value in features.values()):
            print(f"Pominięto komórkę {idx} - wszystkie features równe 0")
            continue
            
        features_list[idx] = features

        predict_class_ml = predict_ml(ml_model['model'], label_encoder, features)
        predicted_classed_ml[idx] = predict_class_ml

        predict_class_cnn = predict_label(cnn_classifier, crop)
        predicted_classes_cnn[idx] = predict_class_cnn[0]

        predicted_clas_fused = predict_fused_func_2(fuse_func, ml_model["model"], label_encoder, cnn_classifier, unet, device, tmp_path)
        predicted_classed_fused[idx] = predicted_clas_fused

        probs_fused = predict_fused_func_2(fuse_func, ml_model["model"], label_encoder, cnn_classifier, unet, device, tmp_path, True)
        
        probs[idx] = {
            'fused': probs_fused
        }
        probs_list.append(probs_fused)
    
    all_indices = set(predicted_classes_cnn.keys()) | set(predicted_classed_ml.keys()) | set(predicted_classed_fused.keys())
    rows = []
    for idx in sorted(all_indices):
        rows.append({
            "idx": idx,
            "predict_vgg": predicted_classes_cnn.get(idx, None),
            "predict_gbm": predicted_classed_ml.get(idx, None),
            "predicted_classed_fused": predicted_classed_fused.get(idx, None),
        })

    df_preds = pd.DataFrame(rows)

    return features_list, predicted_classed_fused, probs, probs_list, df_preds, bbox_image_path, crop_paths

    

