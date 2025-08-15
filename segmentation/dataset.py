import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader

class CellNucleusDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        self.valid_filenames = []
        for name in filenames:
            base = os.path.splitext(name)[0]
            cell_path = os.path.join(mask_dir, base, "cell.png")
            nucleus_path = os.path.join(mask_dir, base, "nucleus.png")
            if os.path.exists(cell_path) and os.path.exists(nucleus_path):
                self.valid_filenames.append(name)
            else:
                print(f"Skipped {name} — no mask.")

    def __len__(self):
        return len(self.valid_filenames)

    def __getitem__(self, idx):
        img_name = self.valid_filenames[idx]
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


image_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\single_all"
mask_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\mask"

all_filenames = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
trainval_filenames, test_filenames = train_test_split(
    all_filenames, test_size=0.05, random_state=42)

train_filenames, val_filenames = train_test_split(
    trainval_filenames, test_size=0.1, random_state=42)

train_dataset = CellNucleusDataset(image_dir, mask_dir, train_filenames)
val_dataset = CellNucleusDataset(image_dir, mask_dir, val_filenames)
test_dataset = CellNucleusDataset(image_dir, mask_dir, test_filenames)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
