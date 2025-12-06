import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
import seaborn as sns
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
import itertools
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
import os
from typing import List, Optional



class CytologyClassifier:
    def __init__(self, num_classes=3, lr=0.0001, device=None, architecture='resnet18', class_counts=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['HSIL', 'LSIL', 'NSIL']
        self.architecture = architecture
        if self.architecture == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            for name, param in self.model.named_parameters():
                if name.startswith('conv1') or name.startswith('bn1') or \
                name.startswith('layer1') or name.startswith('layer2'):
                    param.requires_grad = False 
                else:
                    param.requires_grad = True 
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        elif self.architecture == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            for param in self.model.features.parameters():
                param.requires_grad = False
            for name, param in list(self.model.features.named_parameters())[-8:]:
                param.requires_grad = True
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Linear(25088, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, num_classes)
            )

        elif self.architecture == 'custom_cnn':
            self.model = CNNClassifier(num_classes)

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model = self.model.to(self.device)

        if class_counts:
            total = sum(class_counts.values())
            weights = [total / class_counts[i] for i in range(num_classes)]
            class_weights = torch.tensor(weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, val_loader=None, num_epochs=20, save_best_path=None, early_stopping_patience=10):
        log_dir = f"runs/classification/exp_{self.architecture}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)
        losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_val_f1 = 0
        best_model_state = None
        patience_counter = 0  

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            losses.append(avg_train_loss)
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}", end='', flush=True)

            if val_loader:
                self.model.eval()
                val_running_loss = 0.0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        val_running_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                avg_val_loss = val_running_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                val_f1 = f1_score(all_labels, all_preds, average='macro')
                self.writer.add_scalar("Loss/val", avg_val_loss, epoch)
                self.writer.add_scalar("F1/val", val_f1, epoch)
                print(f" | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0  # reset counter if improvement
                    if save_best_path:
                        torch.save(best_model_state, save_best_path)
                        print(f" 🔥 Best model (F1={val_f1:.4f}) saved to {save_best_path}")
                else:
                    patience_counter += 1
                    print(f" ⏳ No improvement in val loss ({patience_counter}/{early_stopping_patience})")
                    if patience_counter >= early_stopping_patience:
                        print("🛑 Early stopping triggered.")
                        break
            else:
                val_losses.append(None)
            print()

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        # plt.figure(figsize=(8, 4))
        # plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Train Loss')
        # if any(v is not None for v in val_losses):
        #     plt.plot(range(1, len(val_losses) + 1), val_losses, marker='x', label='Val Loss')
        # plt.title('Loss during training')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        self.writer.close()

    def evaluate(self, data_loader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def predict(self, image_path):
        self.model.eval()
        with torch.no_grad():
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_idx = probabilities.argmax().item()
        return self.class_names[predicted_idx]
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)



class CNNClassifier(nn.Module):
    def __init__(self, num_classes, input_size=(224, 224)):
        super(CNNClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),       # 224→220, old version for RGB
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3),      # 220→218
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 218→109

            nn.Conv2d(32, 64, kernel_size=3),      # 109→107
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3),      # 107→105
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 105→52

            nn.Conv2d(64, 128, kernel_size=3),     # 52→50
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3),    # 50→48
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 48→24

            nn.Conv2d(128, 256, kernel_size=3),    # 24→22
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3),    # 22→20
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 20→10
            nn.Dropout(0.3)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_size)  
            dummy_output = self.features(dummy_input)
            flatten_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x





def run_gridsearch_kfold(
    data_dir: str,
    transform,
    architectures: List[str],
    lrs: List[float],
    batch_sizes: List[int],
    num_epochs_list: List[int],
    k_folds: int = 5,
    device: Optional[str] = None,
    output_csv: str = "gridsearch_results.csv"
) -> None:
    """
    Performs k-fold cross-validation with grid search over selected architectures and hyperparameters.
    
    Args:
        data_dir (str): Path to the root directory containing class folders (ImageFolder style).
        transform: Torchvision transform to apply to the images.
        architectures (List[str]): List of model architectures to evaluate (e.g., ['resnet18', 'vgg16']).
        lrs (List[float]): List of learning rates to test.
        batch_sizes (List[int]): List of batch sizes to test.
        num_epochs_list (List[int]): List of epoch counts to train for.
        k_folds (int): Number of folds for Stratified K-Fold CV.
        device (Optional[str]): Device to use ('cuda', 'cpu'). Auto-detected if None.
        output_csv (str): File name for saving grid search results.

    Returns:
        None. Saves results to a CSV file.
    """
    dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    targets = dataset.targets
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    param_grid = list(itertools.product(architectures, lrs, batch_sizes, num_epochs_list))
    print(f"Grid search over {len(param_grid)} combinations...")

    for arch, lr, batch_size, num_epochs in tqdm(param_grid):
        print(f"\nTesting: arch={arch}, lr={lr}, batch_size={batch_size}, epochs={num_epochs}")
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_accuracies = []
        fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(targets)), targets)):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_targets = [targets[i] for i in train_idx]
            class_counts = Counter(train_targets)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = CytologyClassifier(
                num_classes=len(class_names),
                lr=lr,
                architecture=arch,
                class_counts=class_counts,
                device=device
            )

            model.train(train_loader, val_loader, num_epochs=num_epochs)
            acc = model.evaluate(val_loader)

            model.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model.model(x).argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            f1 = f1_score(all_labels, all_preds, average='macro')

            fold_accuracies.append(acc)
            fold_f1s.append(f1)

        result = {
            "architecture": arch,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "mean_accuracy": np.mean(fold_accuracies),
            "std_accuracy": np.std(fold_accuracies),
            "mean_f1": np.mean(fold_f1s),
            "std_f1": np.std(fold_f1s)
        }
        results.append(result)

        print(f"Mean acc: {result['mean_accuracy']:.2f} | F1: {result['mean_f1']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")



def predict_label(classifier, crop_np):
    classifier_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    image_tensor = classifier_transform(crop_np).unsqueeze(0).to(classifier.device)
    with torch.no_grad():
        output = classifier.model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax().item()
        return classifier.class_names[pred_idx], image_tensor
    

