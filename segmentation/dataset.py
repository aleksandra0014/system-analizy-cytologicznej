import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class CellNucleusDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = filenames
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        base_name = os.path.splitext(img_name)[0]

        img_path = os.path.join(self.image_dir, img_name)
        cell_path = os.path.join(self.mask_dir, base_name, "cell.png")
        nucleus_path = os.path.join(self.mask_dir, base_name, "nucleus.png")

        image = Image.open(img_path).convert("RGB")
        cell_mask = Image.open(cell_path).convert("L")
        nucleus_mask = Image.open(nucleus_path).convert("L")

        image = self.transform(image)
        cell_mask = self.transform(cell_mask)[0]
        nucleus_mask = self.transform(nucleus_mask)[0]

        cell_mask = 1.0 - (cell_mask > 0.5).float()
        nucleus_mask = (nucleus_mask > 0.5).float()
        mask = torch.stack([cell_mask, nucleus_mask], dim=0)

        return image, mask

