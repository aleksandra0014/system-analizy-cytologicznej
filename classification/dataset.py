import os
from typing import Counter
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from torchvision.datasets import ImageFolder
from preprocessing.cleaning import clean_image  
from torchvision.datasets import ImageFolder
from PIL import Image

class CleanedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, cleaner=None):
        super().__init__(root, transform=transform)
        self.cleaner = cleaner

    def __getitem__(self, index):
        path, label = self.samples[index]
        if self.cleaner:
            cleaned_np = self.cleaner(path)  # apply custom cleaning
            cleaned_pil = Image.fromarray(cleaned_np)  # convert to PIL
        else:
            cleaned_pil = Image.open(path).convert("RGB")

        if self.transform:
            cleaned_pil = self.transform(cleaned_pil)
        return cleaned_pil, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped'
full_dataset = CleanedImageFolder(root=data_dir, transform=transform, cleaner=clean_image)

dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

targets = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = Counter(targets)

if __name__ == '__main__':
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(class_counts)
