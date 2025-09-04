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

from segmentation.models import UNet, preprocess_image, predict_masks
from segmentation.features import extract_features
from classification.models import CytologyClassifier, predict_label

from llm_testing.helpers import *  # nms_keep_largest_box, predict_gbm, predict_fused_func, predict_vgg_probs, predict_gbm_probs

import matplotlib.pyplot as plt
import matplotlib.patches as patches
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

unet_model_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\models_paths\unet\unet_cell_nucleus_0208.pth"
yolo_model_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\models\yolo_detector_2107_100_20_16_7682\weights\best.pt'
lightgbm_model_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\models_paths\best_model_LightGBM3.pkl'
rf_model_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\models_paths\joined\best_model_RandomForest.pkl"

CLASS_NAMES = ['HSIL', 'LSIL', 'NSIL']
vgg_weights = r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\vgg16\32_0_0001_50_0608.pth'
ARCHITECTURE = 'vgg16'
API_KEY = os.getenv("API_KEY", os.getenv("api_key", ''))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet = UNet(in_channels=3, out_channels=2)
unet.load_state_dict(torch.load(unet_model_path, map_location=device))
unet.to(device).eval()

yolo = YOLO(yolo_model_path)

gbm_model = joblib.load(lightgbm_model_path)
label_encoder = gbm_model['label_encoder']

rf_model = joblib.load(rf_model_path)
rf_encoder = rf_model['label_encoder']

vgg_clf = CytologyClassifier(num_classes=len(CLASS_NAMES), architecture=ARCHITECTURE)
vgg_clf.load(vgg_weights)
vgg_clf.model.eval().to(device)


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

    predict_classes_vgg = {}
    predict_classes_gbm = {}
    predict_fused = {}
    features_list = {}
    probs = {}
    crop_paths = {}  # <--- NEW

    bbox_image_path = None
    if show_image:
        fig, ax = plt.subplots(figsize=(7.68, 5.12), dpi=100)
        ax.imshow(image_np)
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor='#3a5ba0', 
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, max(0, y1 - 5), str(idx),
                color='#f7c873',  
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
            predict_classes_vgg[idx] = 'HSIL/LSIL_group'
            continue

        _, tensor = preprocess_image(tmp_path)
        masks = predict_masks(unet, tensor, device, threshold_nuclei=0.2, threshold_cell=0.5)
        mask_nucleus = cv2.resize(masks[1], (crop.shape[1], crop.shape[0]))
        best_nucleus = select_best_nucleus(mask_nucleus, crop.shape[:2])
        mask_cell = cv2.resize(masks[0], (crop.shape[1], crop.shape[0])) * 255

        features = extract_features(best_nucleus, mask_cell)
        features_list[idx] = features

        predict_class = predict_gbm(gbm_model['model'], label_encoder, features)
        predict_classes_gbm[idx] = predict_class

        predict_class_vgg = predict_label(vgg_clf, crop)
        predict_classes_vgg[idx] = predict_class_vgg[0]

        rf_predictions = predict_fused_func(rf_model['model'], rf_encoder, vgg_clf, unet, device, tmp_path)
        predict_fused[idx] = rf_predictions

        probs_vgg = predict_vgg_probs(vgg_clf, tmp_path, device)
        probs_xgb = predict_gbm_probs(gbm_model['model'], label_encoder, features)
        probs_fused = predict_fused_func(
            rf_model['model'], rf_encoder, vgg_clf, unet, device, tmp_path, probs_output=True
        )
        probs[idx] = {
            'fused': probs_fused
        }

    all_indices = set(predict_classes_vgg.keys()) | set(predict_classes_gbm.keys()) | set(predict_fused.keys())
    rows = []
    for idx in sorted(all_indices):
        rows.append({
            "idx": idx,
            "predict_vgg": predict_classes_vgg.get(idx, None),
            "predict_gbm": predict_classes_gbm.get(idx, None),
            "predict_fused": predict_fused.get(idx, None),
        })

    df_preds = pd.DataFrame(rows)

    return features_list, predict_fused, probs, df_preds, bbox_image_path, crop_paths



if __name__ == "__main__":
    image_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\HSIL\pow 10\35a.bmp"
    features_list, predict_fused, probs, df_preds, bbox_image_path = get_info(image_path, show_image=True)
    print(df_preds)
    print(probs)
    start = datetime.datetime.now()
    response = analyze_with_gemini(bbox_image_path, features_list, predict_fused, probs,
                                   api_key=API_KEY)
                                   # model='llava:7b')
                                    # model='qwen2.5vl:7b', stream=True)
    print(response)  
    end = datetime.datetime.now()
    time = end - start
    print(time)