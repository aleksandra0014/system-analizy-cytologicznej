import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode

class CellNucleusDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, transform=None,
                 allowed_mask_exts=(".png", ".bmp", ".webp")):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.allowed_exts = tuple(e.lower() for e in allowed_mask_exts)

        self.img_tf = transform or T.Compose([
            T.Resize((256, 256)), T.ToTensor()
        ])
        self.mask_tf = T.Compose([
            T.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
            T.ToTensor()  
        ])

        self.valid_filenames = []
        for name in filenames:
            base = os.path.splitext(name)[0]
            cell_path = self._find_mask_path(base, "cell")
            nucleus_path = self._find_mask_path(base, "nucleus")
            if cell_path and nucleus_path:
                self.valid_filenames.append(name)
            else:
                print(f"Skipped {name} — missing mask(s): "
                      f"cell={bool(cell_path)}, nucleus={bool(nucleus_path)}")

    def _find_mask_path(self, base_name: str, stem: str):
        """Zwraca ścieżkę do maski o nazwie stem z dozwolonym rozszerzeniem,
        np. mask_dir/base_name/cell.png|bmp|webp. Jeśli brak — None."""
        base_dir = os.path.join(self.mask_dir, base_name)
        for ext in self.allowed_exts:
            p = os.path.join(base_dir, f"{stem}{ext}")
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.valid_filenames)

    def __getitem__(self, idx):
        img_name = self.valid_filenames[idx]
        base_name = os.path.splitext(img_name)[0]

        img_path = os.path.join(self.image_dir, img_name)
        cell_path = self._find_mask_path(base_name, "cell")
        nucleus_path = self._find_mask_path(base_name, "nucleus")
        if not (cell_path and nucleus_path):
            raise FileNotFoundError(f"Masks not found for {img_name}")

        image = Image.open(img_path).convert("RGB")
        cell_mask = Image.open(cell_path).convert("L")
        nucleus_mask = Image.open(nucleus_path).convert("L")

        image = self.img_tf(image)
        cell_mask = self.mask_tf(cell_mask)[0]
        nucleus_mask = self.mask_tf(nucleus_mask)[0]

        # binarizacja
        cell_mask = (cell_mask > 0.0).float()
        nucleus_mask = (nucleus_mask > 0.0).float()

        mask = torch.stack([cell_mask, nucleus_mask], dim=0) 
        return image, mask


image_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\nowy_unet_train"
mask_dir = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\nowy_unet_train_mask"

image_dir_test = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\nowy_unet_test"
mask_dir_test = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\nowy_unet_test_mask"

all_filenames = [f for f in os.listdir(image_dir)]
train_filenames, val_filenames = train_test_split(
    all_filenames, test_size=0.15, random_state=42)

test_filenames = [f for f in os.listdir(image_dir_test)]

train_dataset = CellNucleusDataset(image_dir, mask_dir, train_filenames)
val_dataset = CellNucleusDataset(image_dir, mask_dir, val_filenames)
test_dataset = CellNucleusDataset(image_dir_test, mask_dir_test, test_filenames)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if __name__ == "__main__":
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    