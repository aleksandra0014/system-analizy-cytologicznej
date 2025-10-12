import os
from PIL import Image
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from models import CytologyClassifier 
from dataset import test_dataset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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
    probs = None

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.device)
            outputs = model.model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            # Zbieranie prawdopodobieństw dla ROC/AUC i PR Curve
            batch_probs = outputs.softmax(dim=1).cpu().numpy()
            if probs is None:
                probs = batch_probs
            else:
                probs = np.vstack([probs, batch_probs])

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
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names, cmap='Purples')
    plt.title(f"Normalized Confusion Matrix - {model_label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(results)
    print(results_df)
    
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ROC/AUC Curve (One-vs-Rest)

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]

    # ROC Curve - fioletowa kolorystyka
    plt.figure(figsize=(7, 5))
    violet_palette = ["#260080", "#E22B53", '#9400D3'] 
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=violet_palette[i % len(violet_palette)], label=f'ROC curve {class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#CCCCCC', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_label}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curve - fioletowa kolorystyka
    plt.figure(figsize=(7, 5))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], probs[:, i])
        plt.plot(recall, precision, lw=2, color=violet_palette[i % len(violet_palette)], label=f'PR curve {class_names[i]} (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_label}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def evaluate_classifier(model_path, test_folder, architecture='vgg16', device=None):
    """
    Evaluate a CytologyClassifier (dowolna architektura) na folderze testowym z podfolderami klas.
    Args:
        model_path (str): Ścieżka do zapisanych wag modelu (pth).
        test_folder (str): Ścieżka do folderu testowego z podfolderami klas.
        architecture (str): Nazwa architektury ('vgg16', 'resnet18', 'custom_cnn').
        device (str, optional): 'cuda' lub 'cpu'.
    """
    classifier = CytologyClassifier(num_classes=3, architecture=architecture, device=device)
    classifier.load(model_path)
    class_names = classifier.class_names

    y_true = []
    y_pred = []

    for subfolder in os.listdir(test_folder):
        true_class = os.path.basename(subfolder)
        subfolder_path = os.path.join(test_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(subfolder_path, filename)
                pred_class = classifier.predict(image_path)
                y_true.append(true_class)
                y_pred.append(pred_class)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average="weighted", zero_division=0
    )

    print("\n=== METRYKI (global) ===")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision (macro):   {prec_macro:.4f}")
    print(f"Recall    (macro):   {rec_macro:.4f}")
    print(f"F1        (macro):   {f1_macro:.4f}")
    print(f"Precision (weighted):{prec_weighted:.4f}")
    print(f"Recall    (weighted):{rec_weighted:.4f}")
    print(f"F1        (weighted):{f1_weighted:.4f}")

    print("\n=== RAPORT (per klasa) ===")
    print(classification_report(y_true, y_pred, labels=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=class_names, normalize='true')
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in class_names],
        columns=[f"pred_{l}" for l in class_names],
    )
    print("\n=== CONFUSION MATRIX (znormalizowana, wartości 0–1) ===")
    print(cm_df)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    ax.set_title("Confusion Matrix (HSIL/LSIL/NSIL, normalized)")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_ylabel("True class")
    ax.set_xlabel("Predicted class")
    fig.tight_layout()
    plt.close()

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

