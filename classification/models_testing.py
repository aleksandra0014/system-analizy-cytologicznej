import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from models import CytologyClassifier 
from dataset import test_dataset

def test_model(
    model_path: str,
    test_loader: torch.utils.data.DataLoader,
    architecture: str,
    model_label: str
) -> None:
    """
    Evaluate a cytology classification model on a test dataset and print results.

    Args:
        model_path (str): Path to the trained model weights.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        architecture (str): Model architecture name (e.g., 'resnet18', 'vgg16').
        model_label (str): Label for the model (used in plots and output).

    Returns:
        None. Prints metrics, confusion matrix, and classification report.
    """
    results = []

    class_names = ['HSIL', 'LSIL', 'NSIL']
    model = CytologyClassifier(num_classes=3, architecture=architecture)
    model.load(model_path)
    model.model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.device)
            outputs = model.model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    results.append({
        "Model": model_label,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    })

    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f"Normalized Confusion Matrix - {model_label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(results)
    print(results_df)
    
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == "__main__":
    test_loader = DataLoader(test_dataset)

    model_info = [
        (r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\resnet18\32_0_001_50_0308.pth', 'ResNet18', 'resnet18'),
        (r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\resnet18\32_0_0001_50_0608.pth', 'ResNet18', 'resnet18'),
        (r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\resnet18\32_0_001_50_0608.pth', 'ResNet18', 'resnet18'),

        (r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\vgg16\16_0_001_50_0308.pth', 'VGG16', 'vgg16'),
        (r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\vgg16\16_0_0001_50_0608.pth', 'VGG16', 'vgg16'),
        (r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\vgg16\32_0_001_50_0208.pth', 'VGG16', 'vgg16'),
        (r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\vgg16\32_0_0001_50_0608.pth', 'VGG16', 'vgg16'),

        (r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\custom_cnn\32_0_001_50_0308.pth', 'CustomCNN', 'custom_cnn')
    ]


    for model_path, model_label, architecture in model_info:
        print(f"\nTestowanie modelu: {model_label, model_path}")
        test_model(model_path, test_loader, architecture, model_label)

