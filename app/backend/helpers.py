
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

import cv2
from PIL import Image
from torchvision import transforms

from segmentation.modelsUnet import  preprocess_image, predict_masks
from segmentation.features import extract_features

FEATURE_NAMES = ['N', 'C', 'NCr', 'Np', 'Cp', 'NCp', 'MinA', 'MinAr', 'MaxA', 'MaxAr',
                 'Nar', 'Car', 'NCar', 'NExt', 'CExt', 'NCExt', 'NSol', 'CSol', 'NCs',
                 'EqN', 'EqC', 'NCEq', 'OrN', 'OrC', 'NCOr']

CLASS_NAMES = ['HSIL', 'LSIL', 'NSIL']

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

def predict_ml(pipe, label_encoder, input_features):
    feature_names = ['N', 'C', 'NCr', 'Np', 'Cp', 'NCp', 'MinA', 'MinAr', 'MaxA', 'MaxAr', 'Nar', 'Car', 'NCar', 'NExt', 'CExt', 'NCExt', 'NSol', 'CSol', 'NCs', 'EqN', 'EqC', 'NCEq', 'OrN', 'OrC', 'NCOr']
    X_new = pd.DataFrame([[input_features[feat] for feat in feature_names]], columns=feature_names)
    y_pred_encoded = pipe.predict(X_new)
    predicted_class = label_encoder.inverse_transform(y_pred_encoded)[0]
    return predicted_class

def preprocess_image_cnn(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return tfm(img)

def predict_cnn_probs(model_or_wrapper, image_path: str, device) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = preprocess_image_cnn(img).unsqueeze(0).to(device)
    net = getattr(model_or_wrapper, "model", model_or_wrapper)
    net.eval()
    logits = net(x)                   
    probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return probs

    
def predict_fused_func_2(fuse_func, lg_model, label_encoder, classifier, unet, device,image_path, probs_output=False):
    pil_image, input_tensor = preprocess_image(image_path)
    predicted_masks = predict_masks(unet, input_tensor, device, threshold_nuclei=0.4, threshold_cell=0.6)
    cell_mask = predicted_masks[0]
    nucleus_mask = predicted_masks[1]
    nucleus_mask = select_best_nucleus(nucleus_mask, nucleus_mask.shape)
    features = extract_features(nucleus_mask, cell_mask)
    feature_names = ['N', 'C', 'NCr', 'Np', 'Cp', 'NCp', 'MinA', 'MinAr', 'MaxA', 'MaxAr', 'Nar', 'Car', 'NCar', 'NExt', 'CExt', 'NCExt', 'NSol', 'CSol', 'NCs', 'EqN', 'EqC', 'NCEq', 'OrN', 'OrC', 'NCOr']
    X_new = pd.DataFrame([[features[feat] for feat in feature_names]], columns=feature_names)
    
    probs2 = predict_ml_probs(lg_model, label_encoder, X_new)

    probs = predict_cnn_probs(classifier, image_path, device)

    
    fused_probs = fuse_func(probs, probs2)

    if not probs_output:
        pred_idx = int(np.argmax(fused_probs))
        pred_label = CLASS_NAMES[pred_idx]
        return pred_label
    else:
        return fused_probs


def predict_ml_probs(model, label_encoder, input_features: dict) -> np.ndarray:
    try:
        X_new = pd.DataFrame([[input_features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    except KeyError as e:
        raise ValueError(f"Brakuje cechy w słowniku: {e}")

    proba = model.predict_proba(X_new)  # <-- tu nie używaj model["model"]
    classes_model = list(label_encoder.classes_)   
    probs_aligned = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for i, cname in enumerate(CLASS_NAMES):
        if cname in classes_model:
            probs_aligned[i] = proba[0, classes_model.index(cname)]
        else:
            probs_aligned[i] = 0.0

    s = probs_aligned.sum()
    if s > 0:
        probs_aligned = probs_aligned / s
    return probs_aligned
