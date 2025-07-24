import os
import numpy as np
import torch
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# === Configuration ===
single_all_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\single_all"
output_mask_root = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\mask"

sam2_checkpoint = r"C:\Users\aleks\OneDrive\Documents\inzynierka\sam2\checkpoints\sam2.1_hiera_large.pt"
model_cfg = r"C:\Users\aleks\OneDrive\Documents\inzynierka\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"

# === Device setup ===
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# === Build SAM2 model and mask generator ===
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    sam2,
    min_mask_region_area=100,
    points_per_side=32
)

# === Make sure output folder exists ===
os.makedirs(output_mask_root, exist_ok=True)

# === Process each image ===
for filename in os.listdir(single_all_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue  # skip non-image files

    image_path = os.path.join(single_all_dir, filename)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Generate masks
    masks = mask_generator.generate(image_np)

    # Create a subfolder for each image under "mask"
    image_name = os.path.splitext(filename)[0]  # remove .jpg/.png
    image_mask_dir = os.path.join(output_mask_root, image_name)
    os.makedirs(image_mask_dir, exist_ok=True)

    # Save each mask as a grayscale PNG
    for i, mask in enumerate(masks):
        mask_img = (mask['segmentation'].astype(np.uint8)) * 255  # convert bool to uint8
        out_path = os.path.join(image_mask_dir, f"mask_{i+1:03d}.png")
        cv2.imwrite(out_path, mask_img)

    print(f"Saved {len(masks)} masks for {filename} in {image_mask_dir}")
