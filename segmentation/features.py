import os
import cv2
import numpy as np
import pandas as pd
import torch
from segmentation.models import UNet, predict_masks, preprocess_image

def get_largest_contour(mask):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None

def compute_region_features(mask):
    mask = (mask > 0).astype(np.uint8)
    contour = get_largest_contour(mask)
    if contour is None:
        return None

    area = np.sum(mask)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0
    extent = area / (w * h) if w * h != 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

    M = cv2.moments(contour)
    if M["mu20"] + M["mu02"] == 0:
        theta = 0
    else:
        theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    orientation = np.tan(2 * theta)


    rect = cv2.minAreaRect(contour)
    (w, h) = rect[1]
    minA = min(w, h)

    hull = cv2.convexHull(contour)
    pts = hull[:, 0, :]
    dmax = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > dmax:
                dmax = d
    maxA = dmax


    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "equivalent_diameter": equivalent_diameter,
        "orientation": orientation,
        "min_axis": minA,
        "max_axis": maxA,
    }


def extract_features(nucleus_mask, cell_mask):
    nucleus = compute_region_features(nucleus_mask)
    cell = compute_region_features(cell_mask)

    if nucleus is None or cell is None:
        return {key: 0 for key in [
            "N", "C", "NCr", "Np", "Cp", "NCp", "MinA", "MinAr", "MaxA", "MaxAr",
            "Nar", "Car", "NCar", "NExt", "CExt", "NCExt", "NSol", "CSol", "NCs",
            "EqN", "EqC", "NCEq", "OrN", "OrC", "NCOr"
        ]}

    features = {}

    # Size
    features["N"] = nucleus["area"]
    features["C"] = cell["area"]
    features["NCr"] = nucleus["area"] / cell["area"] if cell["area"] != 0 else 0

    # Perimeter
    features["Np"] = nucleus["perimeter"]
    features["Cp"] = cell["perimeter"]
    features["NCp"] = nucleus["perimeter"] / cell["perimeter"] if cell["perimeter"] != 0 else 0

    # Axis
    features["MinA"] = nucleus["min_axis"]
    features["MinAr"] = nucleus["min_axis"] / cell["min_axis"] if cell["min_axis"] != 0 else 0
    features["MaxA"] = nucleus["max_axis"]
    features["MaxAr"] = nucleus["max_axis"] / cell["max_axis"] if cell["max_axis"] != 0 else 0

    # Aspect ratio
    features["Nar"] = nucleus["aspect_ratio"]
    features["Car"] = cell["aspect_ratio"]
    features["NCar"] = nucleus["aspect_ratio"] / cell["aspect_ratio"] if cell["aspect_ratio"] != 0 else 0

    # Extent
    features["NExt"] = nucleus["extent"]
    features["CExt"] = cell["extent"]
    features["NCExt"] = nucleus["extent"] / cell["extent"] if cell["extent"] != 0 else 0

    # Solidity
    features["NSol"] = nucleus["solidity"]
    features["CSol"] = cell["solidity"]
    features["NCs"] = nucleus["solidity"] / cell["solidity"] if cell["solidity"] != 0 else 0

    # Equivalent diameter
    features["EqN"] = nucleus["equivalent_diameter"]
    features["EqC"] = cell["equivalent_diameter"]
    features["NCEq"] = nucleus["equivalent_diameter"] / cell["equivalent_diameter"] if cell["equivalent_diameter"] != 0 else 0

    # Orientation
    features["OrN"] = nucleus["orientation"]
    features["OrC"] = cell["orientation"]
    features["NCOr"] = nucleus["orientation"] / cell["orientation"] if cell["orientation"] != 0 else 0

    return features


if __name__ == "__main__":

    model = UNet(in_channels=3, out_channels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(
        r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\unet_cell_nucleus_0208.pth",
        map_location=device
    ))
    model.to(device)
    model.eval() 

    base_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped3"
    splits = ["train", "test", "val"]

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Brak katalogu: {split_dir}")
            continue

        rows = []
        processed = 0

        for subfolder in os.listdir(split_dir):
            subfolder_path = os.path.join(split_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for fname in os.listdir(subfolder_path):
                if not fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(subfolder_path, fname)

                try:

                    pil_image, input_tensor = preprocess_image(image_path)

                    if input_tensor is None or input_tensor.ndim != 4:
                        raise ValueError("preprocess_image zwrócił zły input_tensor")

                    input_tensor = input_tensor.to(device)

                    with torch.no_grad(): 
                        predicted_masks = predict_masks(
                            model,
                            input_tensor,
                            device,
                            threshold_nuclei=0.2,  
                            threshold_cell=0.5
                        )

                    cell_mask = (predicted_masks[0] > 0).astype(np.uint8)
                    nucleus_mask = (predicted_masks[1] > 0).astype(np.uint8)

                    cpos = int(cell_mask.sum())
                    npos = int(nucleus_mask.sum())
                    if cpos == 0 or npos == 0:
                        print(f"[INFO] Pusta maska w {image_path} -> cell:{cpos} nucleus:{npos}")

                    features = extract_features(nucleus_mask, cell_mask)

                except Exception as e:
                    import traceback
                    print(f"[ERROR] {image_path}: {e}")
                    traceback.print_exc()
                    continue 

                features["class"] = subfolder

                rows.append(features)
                processed += 1

        # Zapis CSV tylko jeśli coś przetworzono
        out_csv = os.path.join(base_dir, f"features_{split}.csv")
        if len(rows) == 0:
            print(f"[WARN] Brak wierszy dla splitu '{split}'. Nie zapisuję CSV.")
        else:
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)
            print(f"[OK] Zapisano {out_csv} z {len(df)} wierszami (przetworzono {processed} plików).")

    