import os
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as TF


def augment_image(image):
    """
    Apply a series of augmentations to the input image.
    """
    augment = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=50, fill=(255, 255, 255)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    return augment(image)

if __name__ == "__main__":
    augment = augment_image  

    input_root = r"\data\data_single_cropped"      
    output_root = r"\data\data_single_cropped"      
    num_augments = 2 

    for cls_name in os.listdir(input_root):
        cls_path = os.path.join(input_root, cls_name)
        if not os.path.isdir(cls_path):
            continue

        output_cls_path = os.path.join(output_root, cls_name)
        os.makedirs(output_cls_path, exist_ok=True)

        for fname in os.listdir(cls_path):
            if not fname.lower().endswith((".jpg", ".png", ".bmp")):
                continue

            img_path = os.path.join(cls_path, fname)
            img = Image.open(img_path).convert("RGB")

            for i in range(num_augments):
                aug_img = augment(img)
                aug_fname = f"{os.path.splitext(fname)[0]}_aug{i}.png"
                aug_img_pil = TF.to_pil_image(aug_img)
                aug_img_pil.save(os.path.join(output_cls_path, aug_fname))
