import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_nucleus_and_cell_contours(image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Extracts the nucleus and cell contours from an image file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (image_rgb, mask_nucleus, nucleus_contour_img, cell_mask, cell_contour)
            image_rgb (np.ndarray): The RGB image.
            mask_nucleus (np.ndarray): Binary mask of the nucleus.
            nucleus_contour_img (np.ndarray): Edge image of the nucleus contour.
            cell_mask (np.ndarray): Binary mask of the cell.
            cell_contour (np.ndarray): Edge image of the cell contour.
        None: If the image cannot be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    white_mask = cv2.inRange(image_rgb, np.array([220, 220, 220]), np.array([255, 255, 255]))  # removing background
    not_white_mask = cv2.bitwise_not(white_mask)

    gray_nonwhite = gray[not_white_mask > 0]
    if len(gray_nonwhite) == 0:
        return image_rgb, np.zeros_like(gray), np.zeros_like(gray), np.zeros_like(gray), np.zeros_like(gray)

    mean_intensity = np.mean(gray_nonwhite)  # mean intensity of non-white pixels
    dynamic_thresh = int(mean_intensity * 0.8)  # dynamic threshold based on mean intensity to detect dark nucleus

    dark_mask = cv2.inRange(gray, 0, dynamic_thresh)  # mask for dark areas
    dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=not_white_mask)

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nucleus_clean = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours_nucleus, _ = cv2.findContours(nucleus_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_nucleus = np.zeros_like(gray)
    nucleus_contour_img = np.zeros_like(gray)

    # Choose the contour with the lowest intensity
    if contours_nucleus:
        min_intensity = 999
        selected_contour = None

        for cnt in contours_nucleus:
            temp_mask = np.zeros_like(gray)
            cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
            mean_val = np.mean(gray[temp_mask == 255])

            if mean_val < min_intensity:
                min_intensity = mean_val
                selected_contour = cnt

        if selected_contour is not None:
            cv2.drawContours(mask_nucleus, [selected_contour], -1, 255, -1)
            nucleus_contour_img = cv2.Canny(mask_nucleus, 50, 150)

    cell_mask = cv2.bitwise_not(white_mask)
    cell_blurred = cv2.GaussianBlur(cell_mask, (3, 3), 0)
    cell_contour = cv2.Canny(cell_blurred, 30, 100)
    cell_mask = cv2.bitwise_not(white_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, kernel)
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, kernel)

    return image_rgb, mask_nucleus, nucleus_contour_img, cell_mask, cell_contour

def get_mean_ratio_file(file_path):
    """
    Calculates the mean nucleus-to-cell area ratio for a given image file.

    Args:
        file_path (str): Path to the image file.

    Returns:
        float: The ratio (in percent) of nucleus area to cell area.
    """
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    result = extract_nucleus_and_cell_contours(file_path)

    if result is None:
        print(f" (nieprawidłowy obraz)")

    image, nucleus_mask, nucleus_contour, cell_mask, cell_contour = result
    nucleus_area = np.count_nonzero(nucleus_mask)
    cell_area = np.count_nonzero(cell_mask)

    ratio = 100 * nucleus_area / cell_area
    return ratio

NSIL_range = (float(6.028023671609743 - 8.003730131911812), float(6.028023671609743 + 8.003730131911812))
LSIL_range = (float(17.960424045561066 - 11.085963449966334), float(17.960424045561066 + 11.085963449966334))
HSIL_range = (float(35.708751454451765 - 10.645653397929129), float(35.708751454451765 + 10.645653397929129))
