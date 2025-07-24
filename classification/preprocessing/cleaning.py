import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def clean_image(image_path: str) -> np.ndarray:
    """
    Denoises and normalizes a color image.
    Returns normalized uint8 image.
    """
    img = cv2.imread(image_path)

    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    normalized = denoised / 255.0
    normalized_visual = (normalized * 255).astype(np.uint8)

    return normalized_visual
