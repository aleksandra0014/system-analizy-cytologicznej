import cv2
import numpy as np
from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image

def extract_features(nucleus_mask, cell_mask):
    features = {}

    # Upewnij się, że maski są binarne
    nucleus_mask = (nucleus_mask > 0).astype(np.uint8)
    cell_mask = (cell_mask > 0).astype(np.uint8)

    # Region properties
    nucleus_props = regionprops(label(nucleus_mask))[0]
    cell_props = regionprops(label(cell_mask))[0]

    # Pole powierzchni (area)
    features['NucleusArea'] = nucleus_props.area
    features['CellArea'] = cell_props.area
    features['AreaRatio'] = nucleus_props.area / cell_props.area if cell_props.area != 0 else 0

    # Obwód (perimeter) – długość konturu
    contours_n = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_c = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    features['Np'] = cv2.arcLength(contours_n[0], True)
    features['Cp'] = cv2.arcLength(contours_c[0], True)
    features['NCp'] = features['Np'] / features['Cp'] if features['Cp'] != 0 else 0

    # MinA i MaxA – ekstremalne odległości punktów
    def min_max_axis(mask):
        points = np.column_stack(np.where(mask > 0))
        if len(points) < 2:
            return 0, 0
        dists = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        return np.min(dists), np.max(dists)

    nucleus_MinA, nucleus_MaxA = min_max_axis(nucleus_mask)
    cell_MinA, cell_MaxA = min_max_axis(cell_mask)
    features['MinA'] = nucleus_MinA
    features['MinAr'] = nucleus_MinA / cell_MinA if cell_MinA != 0 else 0
    features['MaxA'] = nucleus_MaxA
    features['MaxAr'] = nucleus_MaxA / cell_MaxA if cell_MaxA != 0 else 0

    # Bounding box dimensions
    BB_n = nucleus_props.bbox
    BB_c = cell_props.bbox
    BBw_n = BB_n[3] - BB_n[1]
    BBh_n = BB_n[2] - BB_n[0]
    BBw_c = BB_c[3] - BB_c[1]
    BBh_c = BB_c[2] - BB_c[0]

    # Aspect Ratio
    Nar = BBw_n / BBh_n if BBh_n != 0 else 0
    Car = BBw_c / BBh_c if BBh_c != 0 else 0
    features['Nar'] = Nar
    features['Car'] = Car
    features['NCar'] = Nar / Car if Car != 0 else 0

    # Extent
    features['NExt'] = nucleus_props.area / (BBw_n * BBh_n) if BBw_n * BBh_n != 0 else 0
    features['CExt'] = cell_props.area / (BBw_c * BBh_c) if BBw_c * BBh_c != 0 else 0
    features['NCExt'] = features['NExt'] / features['CExt'] if features['CExt'] != 0 else 0

    # Solidity (wypukłość)
    nucleus_ch = convex_hull_image(nucleus_mask)
    cell_ch = convex_hull_image(cell_mask)
    NSol = nucleus_props.area / np.sum(nucleus_ch)
    CSol = cell_props.area / np.sum(cell_ch)
    features['NSol'] = NSol
    features['CSol'] = CSol
    features['NCs'] = NSol / CSol if CSol != 0 else 0

    # Equivalence diameter
    features['EqN'] = nucleus_props.equivalent_diameter
    features['EqC'] = cell_props.equivalent_diameter
    features['NCEq'] = features['EqN'] / features['EqC'] if features['EqC'] != 0 else 0

    # Orientation
    features['OrN'] = np.tan(2 * nucleus_props.orientation)
    features['OrC'] = np.tan(2 * cell_props.orientation)
    features['NCOr'] = features['OrN'] / features['OrC'] if features['OrC'] != 0 else 0

    return features


if __name__ == "__main__":
    from models import *
    model = UNet(n_channels=3, n_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("unet_cell_nucleus_0208_40_clahe.pth"))
    model.to(device)
    image_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\single_all_clahe\2b_1.jpg"
    pil_image, input_tensor = preprocess_image(image_path)
    predicted_masks = predict_masks(model, input_tensor, device, threshold_nuclei=0.3, threshold_cell=0.5)
    cell_mask = predicted_masks[0]
    nucleus_mask = predicted_masks[1]
    features = extract_features(nucleus_mask, cell_mask)
    print(features)