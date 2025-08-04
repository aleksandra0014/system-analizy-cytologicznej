import os
from typing import Counter
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from torchvision.datasets import ImageFolder
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data_dir = r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped3\train'
train_dataset = ImageFolder(root=train_data_dir, transform=transform)

val_data_dir = r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped3\val'
val_dataset = ImageFolder(root=val_data_dir, transform=transform)

test_data_dir = r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped3\test'
test_dataset = ImageFolder(root=test_data_dir, transform=transform)

class_counts = Counter(train_dataset.targets)

if __name__ == '__main__':
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print("Train class distribution:")
    train_class_counts = Counter(train_dataset.targets)
    for idx, class_name in enumerate(train_dataset.classes):
        print(f"  {class_name}: {train_class_counts[idx]}")
    val_class_counts = Counter(val_dataset.targets)
    test_class_counts = Counter(test_dataset.targets)

    print("Val class distribution:")
    for idx, class_name in enumerate(val_dataset.classes):
        print(f"  {class_name}: {val_class_counts[idx]}")

    print("Test class distribution:")
    for idx, class_name in enumerate(test_dataset.classes):
        print(f"  {class_name}: {test_class_counts[idx]}")

