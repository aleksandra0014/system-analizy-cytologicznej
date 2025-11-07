import os
import numpy as np
import torch
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

single_all_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\p"
output_mask_root = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\p_mask"

sam2_checkpoint = r"C:\Users\aleks\OneDrive\Documents\inzynierka\sam2\checkpoints\sam2.1_hiera_large.pt"
model_cfg = r"C:\Users\aleks\OneDrive\Documents\inzynierka\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    sam2,
    min_mask_region_area=100,
    points_per_side=32
)

os.makedirs(output_mask_root, exist_ok=True)

for filename in os.listdir(single_all_dir):
    image_path = os.path.join(single_all_dir, filename)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    masks = mask_generator.generate(image_np)

    image_name = os.path.splitext(filename)[0]  
    image_mask_dir = os.path.join(output_mask_root, image_name)
    os.makedirs(image_mask_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        mask_img = (mask['segmentation'].astype(np.uint8)) * 255  
        out_path = os.path.join(image_mask_dir, f"mask_{i+1:03d}.png")
        cv2.imwrite(out_path, mask_img)

    print(f"Saved {len(masks)} masks for {filename} in {image_mask_dir}")
