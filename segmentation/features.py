import cv2
import numpy as np
from skimage.morphology import convex_hull_image


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

    # Używamy tylko punktów z konturu (a nie całej maski!)
    points = contour[:, 0, :]
    if len(points) < 2:
        min_axis = max_axis = 0
    else:
        # MaxA: największa odległość między punktami konturu
        dmax = 0
        dmin = np.inf
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d > dmax:
                    dmax = d
                if d < dmin:
                    dmin = d
        max_axis = dmax
        min_axis = dmin

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "equivalent_diameter": equivalent_diameter,
        "orientation": orientation,
        "min_axis": min_axis,
        "max_axis": max_axis,
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
    from models import *
    model = UNet(in_channels=3, out_channels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\unet_cell_nucleus_0208_40_clahe.pth"))
    model.to(device)
    image_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped2\HSIL\51b_1.bmp"
    pil_image, input_tensor = preprocess_image(image_path)
    predicted_masks = predict_masks(model, input_tensor, device, threshold_nuclei=0.3, threshold_cell=0.5)
    cell_mask = predicted_masks[0]
    nucleus_mask = predicted_masks[1]
    features = extract_features(nucleus_mask, cell_mask)
    print(features)