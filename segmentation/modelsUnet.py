
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# # class DoubleConv(nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super().__init__()
# #         self.double_conv = nn.Sequential(
# #             nn.Conv2d(in_channels, out_channels, 3, padding=1),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(out_channels, out_channels, 3, padding=1),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True)
# #         )

# #     def forward(self, x):
# #         return self.double_conv(x)

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):  # 2 wyjścia: komórka i jądro
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.output_layer = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))

        return self.output_layer(dec1)
    
class UNet4Levels(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        
        # Encoder (Contracting Path)
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)  # 1024 bo concat z encoder4
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = DoubleConv(512, 256)   # 512 bo concat z encoder3
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = DoubleConv(256, 128)   # 256 bo concat z encoder2
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = DoubleConv(128, 64)    # 128 bo concat z encoder1
        
        # Output Layer
        self.output_layer = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)           # 64 kanałów
        enc2 = self.encoder2(self.pool1(enc1))  # 128 kanałów
        enc3 = self.encoder3(self.pool2(enc2))  # 256 kanałów
        enc4 = self.encoder4(self.pool3(enc3))  # 512 kanałów
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # 1024 kanały
        
        # Decoder z skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Concat po wymiarze kanałów
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output
        return self.output_layer(dec1)

def evaluate(model, val_loader, loss_fn, device, writer=None, step=None):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if writer and step is not None:
            preds = torch.sigmoid(preds)

            writer.add_scalar("Loss/val", avg_val_loss, step)
            writer.add_images("Val/Input", images, step)
            writer.add_images("Val/GroundTruth/cell", masks[:, 0:1], step)
            writer.add_images("Val/GroundTruth/nucleus", masks[:, 1:2], step)
            writer.add_images("Val/Prediction/cell", preds[:, 0:1], step)
            writer.add_images("Val/Prediction/nucleus", preds[:, 1:2], step)

    model.train()
    return avg_val_loss



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (inputs * targets).sum(-1)
        union = inputs.sum(-1) + targets.sum(-1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

def combined_loss(preds, targets):
    return 0.5 * bce_loss(preds, targets) + 0.5 * dice_loss(preds, targets)


def train(model, train_loader, val_loader, optimizer, device, epochs=10,
          log_dir="runs/exp1", model_path="best_model.pth"):
    
    model.to(device)
    writer = SummaryWriter(log_dir)
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks)  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)

            if global_step % 100 == 0:
                with torch.no_grad():
                    preds = torch.sigmoid(outputs)
                    writer.add_images("Train/Input", images, global_step)
                    writer.add_images("Train/GroundTruth/cell", masks[:, 0:1], global_step)
                    writer.add_images("Train/GroundTruth/nucleus", masks[:, 1:2], global_step)
                    writer.add_images("Train/Prediction/cell", preds[:, 0:1], global_step)
                    writer.add_images("Train/Prediction/nucleus", preds[:, 1:2], global_step)

            global_step += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train loss: {avg_epoch_loss:.4f}")

        val_loss = evaluate(model, val_loader, combined_loss, device, writer, global_step)
        print(f"Epoch {epoch+1} Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved (val loss = {val_loss:.4f})")

    writer.close()

def dice_score(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (2. * intersection + epsilon) / (union + epsilon)

def iou_score(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum(dim=(1, 2))
    union = ((pred + target) >= 1).float().sum(dim=(1, 2))
    return (intersection + epsilon) / (union + epsilon)

def precision(pred, target, epsilon=1e-6):
    tp = (pred * target).sum(dim=(1, 2))
    fp = (pred * (1 - target)).sum(dim=(1, 2))
    return (tp + epsilon) / (tp + fp + epsilon)

def recall(pred, target, epsilon=1e-6):
    tp = (pred * target).sum(dim=(1, 2))
    fn = ((1 - pred) * target).sum(dim=(1, 2))
    return (tp + epsilon) / (tp + fn + epsilon)


def test_model(model, test_loader, device, threshold_cell=0.5, threshold_nucleus=0.5):
    """
    Testuje model segmentacyjny na zbiorze testowym, używając
    oddzielnych progów dla maski komórki (kanał 0) i jądra (kanał 1).
    
    Oblicza metryki osobno dla każdej klasy oraz uśrednione (macro-average).
    """
    model.eval()
    model.to(device)

    dice_cell_all = []
    iou_cell_all = []
    precision_cell_all = []
    recall_cell_all = []

    dice_nucleus_all = []
    iou_nucleus_all = []
    precision_nucleus_all = []
    recall_nucleus_all = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device) # Oczekiwany kształt: [B, 2, H, W]

            outputs = torch.sigmoid(model(images)) # Kształt: [B, 2, H, W]

            # Kanał 0: Komórka
            preds_cell = (outputs[:, 0] > threshold_cell).float()
            masks_cell = masks[:, 0]
            
            # Kanał 1: Jądro
            preds_nucleus = (outputs[:, 1] > threshold_nucleus).float()
            masks_nucleus = masks[:, 1]

            # Oblicz i zapisz metryki dla komórki (kanał 0)
            dice_cell_all.append(dice_score(preds_cell, masks_cell))
            iou_cell_all.append(iou_score(preds_cell, masks_cell))
            precision_cell_all.append(precision(preds_cell, masks_cell))
            recall_cell_all.append(recall(preds_cell, masks_cell))
            
            # Oblicz i zapisz metryki dla jądra (kanał 1)
            dice_nucleus_all.append(dice_score(preds_nucleus, masks_nucleus))
            iou_nucleus_all.append(iou_score(preds_nucleus, masks_nucleus))
            precision_nucleus_all.append(precision(preds_nucleus, masks_nucleus))
            recall_nucleus_all.append(recall(preds_nucleus, masks_nucleus))


    def mean_metric(metric_list):
        if not metric_list:
            return 0.0
        all_vals = torch.cat(metric_list, dim=0) 
        return all_vals.mean().item() 

    avg_dice_cell = mean_metric(dice_cell_all)
    avg_iou_cell = mean_metric(iou_cell_all)
    avg_precision_cell = mean_metric(precision_cell_all)
    avg_recall_cell = mean_metric(recall_cell_all)

    avg_dice_nucleus = mean_metric(dice_nucleus_all)
    avg_iou_nucleus = mean_metric(iou_nucleus_all)
    avg_precision_nucleus = mean_metric(precision_nucleus_all)
    avg_recall_nucleus = mean_metric(recall_nucleus_all)

    avg_dice_total = (avg_dice_cell + avg_dice_nucleus) / 2
    avg_iou_total = (avg_iou_cell + avg_iou_nucleus) / 2
    avg_precision_total = (avg_precision_cell + avg_precision_nucleus) / 2
    avg_recall_total = (avg_recall_cell + avg_recall_nucleus) / 2

    print("\n" + "="*40)
    print(" " * 12 + "WYNIKI EWALUACJI")
    print("="*40)

    print("\n📊 --- Wyniki dla Komórki (Kanał 0) ---")
    print(f"Próg:     {threshold_cell}")
    print(f"Dice:     {avg_dice_cell:.4f}")
    print(f"IoU:      {avg_iou_cell:.4f}")
    print(f"Precision:{avg_precision_cell:.4f}")
    print(f"Recall:   {avg_recall_cell:.4f}")

    print("\n🔬 --- Wyniki dla Jądra (Kanał 1) ---")
    print(f"Próg:     {threshold_nucleus}")
    print(f"Dice:     {avg_dice_nucleus:.4f}")
    print(f"IoU:      {avg_iou_nucleus:.4f}")
    print(f"Precision:{avg_precision_nucleus:.4f}")
    print(f"Recall:   {avg_recall_nucleus:.4f}")

    print("\n📈 --- Wyniki Uśrednione (Macro-Average) ---")
    print(f"Mean Dice:     {avg_dice_total:.4f}")
    print(f"Mean IoU:      {avg_iou_total:.4f}")
    print(f"Mean Precision:{avg_precision_total:.4f}")
    print(f"Mean Recall:   {avg_recall_total:.4f}")
    print("="*40)

def load_model(weights_path, device):
    model = UNet(in_channels=3, out_channels=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  
    return image, tensor

def predict_masks(model, image_tensor, device, threshold_nuclei=0.3, threshold_cell=0.7):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        preds = torch.sigmoid(output).squeeze(0)  # [2, H, W]

        mask_cell = (preds[0] > threshold_cell).float().cpu().numpy()  # jądro
        mask_nucleus = (preds[1] > threshold_nuclei).float().cpu().numpy()      # komórka

        # mask_cell = postprocess_mask(mask_cell)
        # mask_nucleus = postprocess_mask(mask_nucleus)
        # mask_nucleus = select_best_nucleus(mask_nucleus, image_tensor.shape[2:])

    return [mask_cell, mask_nucleus]  

def select_best_nucleus(mask, image_shape):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    h, w = image_shape
    center = np.array([w // 2, h // 2])
    best_score, best_contour = -np.inf, None
    for cnt in contours:
        if len(cnt) < 5: continue
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), _ = ellipse
        if MA == 0 or ma == 0: continue
        owalnosc = min(MA, ma) / max(MA, ma)
        dist_to_center = np.linalg.norm(np.array([x, y]) - center)
        score = owalnosc / (dist_to_center + 1e-5)
        if score > best_score:
            best_score, best_contour = score, cnt
    out = np.zeros_like(mask, dtype=np.uint8)
    if best_contour is not None:
        cv2.drawContours(out, [best_contour], -1, 255, -1)
    return out

def test_model_with_thresholds(model, test_loader, device, 
                                threshold_cell_range=[0.3, 0.4, 0.5, 0.6, 0.7],
                                threshold_nucleus_range=[0.2, 0.3, 0.4, 0.5, 0.6]):
    """
    Testuje model z różnymi progami dla komórek i jąder
    
    Args:
        model: model U-Net
        test_loader: DataLoader z danymi testowymi
        device: urządzenie (cuda/cpu)
        threshold_cell_range: lista progów do testowania dla komórek
        threshold_nucleus_range: lista progów do testowania dla jąder
    
    Returns:
        dict: wyniki dla każdej kombinacji progów
    """
    model.eval()
    model.to(device)
    
    results = {}
    
    # Zbierz wszystkie predykcje i ground truth
    all_outputs = []
    all_masks = []
    
    print("📥 Collecting predictions...")
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = torch.sigmoid(model(images))
            all_outputs.append(outputs.cpu())
            all_masks.append(masks.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)  # [N, 2, H, W]
    all_masks = torch.cat(all_masks, dim=0)      # [N, 2, H, W]
    
    print(f"🔬 Testing {len(threshold_cell_range)} x {len(threshold_nucleus_range)} threshold combinations...\n")
    
    # Testuj każdą kombinację progów
    for thresh_cell in threshold_cell_range:
        for thresh_nucleus in threshold_nucleus_range:
            # Zastosuj progi
            preds_cell = (all_outputs[:, 0] > thresh_cell).float()
            preds_nucleus = (all_outputs[:, 1] > thresh_nucleus).float()
            
            # Oblicz metryki dla komórek (kanał 0)
            dice_cell = dice_score(preds_cell, all_masks[:, 0]).mean()
            iou_cell = iou_score(preds_cell, all_masks[:, 0]).mean()
            precision_cell = precision(preds_cell, all_masks[:, 0]).mean()
            recall_cell = recall(preds_cell, all_masks[:, 0]).mean()
            
            # Oblicz metryki dla jąder (kanał 1)
            dice_nucleus = dice_score(preds_nucleus, all_masks[:, 1]).mean()
            iou_nucleus = iou_score(preds_nucleus, all_masks[:, 1]).mean()
            precision_nucleus = precision(preds_nucleus, all_masks[:, 1]).mean()
            recall_nucleus = recall(preds_nucleus, all_masks[:, 1]).mean()
            
            # Średnie metryki
            avg_dice = (dice_cell + dice_nucleus) / 2
            avg_iou = (iou_cell + iou_nucleus) / 2
            avg_precision = (precision_cell + precision_nucleus) / 2
            avg_recall = (recall_cell + recall_nucleus) / 2
            
            # Zapisz wyniki
            key = (thresh_cell, thresh_nucleus)
            results[key] = {
                'cell': {
                    'dice': dice_cell.item(),
                    'iou': iou_cell.item(),
                    'precision': precision_cell.item(),
                    'recall': recall_cell.item()
                },
                'nucleus': {
                    'dice': dice_nucleus.item(),
                    'iou': iou_nucleus.item(),
                    'precision': precision_nucleus.item(),
                    'recall': recall_nucleus.item()
                },
                'average': {
                    'dice': avg_dice.item(),
                    'iou': avg_iou.item(), 
                    'precision': avg_precision.item(),
                    'recall': avg_recall.item()
                }
            }
    
    return results


def find_best_thresholds(results, metric='dice'):
    """
    Znajduje najlepsze progi na podstawie wybranej metryki
    
    Args:
        results: dict z wynikami z test_model_with_thresholds
        metric: metryka do optymalizacji ('dice', 'iou', 'precision', 'recall')
    
    Returns:
        dict: najlepsze progi i ich wyniki
    """
    best_score = -1
    best_thresholds = None
    best_results = None
    print(results)
    for (thresh_cell, thresh_nucleus), res in results.items():
        score = res['average'][metric]
        if score > best_score:
            best_score = score
            best_thresholds = (thresh_cell, thresh_nucleus)
            best_results = res
    
    return {
        'thresholds': best_thresholds,
        'results': best_results,
        'score': best_score
    }


def print_threshold_results(results, top_n=5):
    """
    Wyświetla top N najlepszych kombinacji progów
    """
    # Sortuj według średniego Dice score
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['average']['dice'], 
                           reverse=True)
    
    print(f"\n🏆 Top {top_n} Threshold Combinations (by Average Dice):\n")
    print("=" * 90)
    print(f"{'Rank':<5} {'Cell Thr':<10} {'Nucleus Thr':<12} {'Avg Dice':<10} {'Avg IoU':<10} {'Cell Dice':<10} {'Nuc Dice':<10}")
    print("=" * 90)
    
    for i, ((thresh_cell, thresh_nucleus), res) in enumerate(sorted_results[:top_n], 1):
        print(f"{i:<5} {thresh_cell:<10.2f} {thresh_nucleus:<12.2f} "
              f"{res['average']['dice']:<10.4f} {res['average']['iou']:<10.4f} "
              f"{res['cell']['dice']:<10.4f} {res['nucleus']['dice']:<10.4f}")
    
    print("=" * 90)


def plot_threshold_heatmap(results, metric='dice'):
    """
    Wizualizuje wyniki jako heatmapy dla komórek i jąder
    """
    # Wyciągnij unikalne progi
    thresholds = list(results.keys())
    cell_thresholds = sorted(list(set([t[0] for t in thresholds])))
    nucleus_thresholds = sorted(list(set([t[1] for t in thresholds])))
    
    # Przygotuj macierze wyników
    cell_matrix = np.zeros((len(nucleus_thresholds), len(cell_thresholds)))
    nucleus_matrix = np.zeros((len(nucleus_thresholds), len(cell_thresholds)))
    avg_matrix = np.zeros((len(nucleus_thresholds), len(cell_thresholds)))
    
    for (thresh_cell, thresh_nucleus), res in results.items():
        i = nucleus_thresholds.index(thresh_nucleus)
        j = cell_thresholds.index(thresh_cell)
        cell_matrix[i, j] = res['cell'][metric]
        nucleus_matrix[i, j] = res['nucleus'][metric]
        avg_matrix[i, j] = res['average'][metric]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = axes[0].imshow(cell_matrix, cmap='YlOrRd', aspect='auto')
    axes[0].set_title(f'Cell {metric.capitalize()}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Cell Threshold')
    axes[0].set_ylabel('Nucleus Threshold')
    axes[0].set_xticks(range(len(cell_thresholds)))
    axes[0].set_yticks(range(len(nucleus_thresholds)))
    axes[0].set_xticklabels([f'{t:.2f}' for t in cell_thresholds])
    axes[0].set_yticklabels([f'{t:.2f}' for t in nucleus_thresholds])
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(nucleus_matrix, cmap='YlGnBu', aspect='auto')
    axes[1].set_title(f'Nucleus {metric.capitalize()}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cell Threshold')
    axes[1].set_ylabel('Nucleus Threshold')
    axes[1].set_xticks(range(len(cell_thresholds)))
    axes[1].set_yticks(range(len(nucleus_thresholds)))
    axes[1].set_xticklabels([f'{t:.2f}' for t in cell_thresholds])
    axes[1].set_yticklabels([f'{t:.2f}' for t in nucleus_thresholds])
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(avg_matrix, cmap='viridis', aspect='auto')
    axes[2].set_title(f'Average {metric.capitalize()}', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Cell Threshold')
    axes[2].set_ylabel('Nucleus Threshold')
    axes[2].set_xticks(range(len(cell_thresholds)))
    axes[2].set_yticks(range(len(nucleus_thresholds)))
    axes[2].set_xticklabels([f'{t:.2f}' for t in cell_thresholds])
    axes[2].set_yticklabels([f'{t:.2f}' for t in nucleus_thresholds])
    plt.colorbar(im3, ax=axes[2])
    
    # Zaznacz najlepsze wartości
    best_idx = np.unravel_index(avg_matrix.argmax(), avg_matrix.shape)
    axes[2].plot(best_idx[1], best_idx[0], 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    
    plt.tight_layout()
    plt.show()


def compare_thresholds_visually(model, image_path, device, threshold_combinations):
    """
    Porównuje wizualizacje dla różnych kombinacji progów
    """
    pil_image, input_tensor = preprocess_image(image_path)
    
    n_combinations = len(threshold_combinations)
    fig, axes = plt.subplots(n_combinations, 3, figsize=(12, 4 * n_combinations))
    
    if n_combinations == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (thresh_cell, thresh_nucleus) in enumerate(threshold_combinations):
        predicted_masks = predict_masks(model, input_tensor, device, thresh_nucleus, thresh_cell)
        cell_mask = predicted_masks[0]
        nucleus_mask = predicted_masks[1]
        
        axes[idx, 0].imshow(pil_image)
        axes[idx, 0].set_title(f"Original\nCell: {thresh_cell}, Nucleus: {thresh_nucleus}")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cell_mask, cmap='gray')
        axes[idx, 1].set_title(f"Cell Mask (t={thresh_cell})")
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(nucleus_mask, cmap='gray')
        axes[idx, 2].set_title(f"Nucleus Mask (t={thresh_nucleus})")
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_all_images_in_folder(model, device, folder_path, num_images=None, threshold_nuclei=0.2, threshold_cell=0.5, select_nucleus=False):
    import os
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if num_images is not None:
        image_files = image_files[:num_images]
    n = len(image_files)
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axs = [axs]
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        pil_image, input_tensor = preprocess_image(image_path)
        predicted_masks = predict_masks(model, input_tensor, device, threshold_nuclei, threshold_cell)
        cell_mask = predicted_masks[0]
        nucleus_mask = predicted_masks[1]
        if select_nucleus:
            nucleus_mask = select_best_nucleus(nucleus_mask, pil_image.size[::-1])
        axs[idx][0].imshow(pil_image)
        axs[idx][0].set_title(f"Oryginal: {filename}")
        axs[idx][0].axis("off")
        axs[idx][1].imshow(cell_mask, cmap='gray')
        axs[idx][1].set_title("Mask: cell")
        axs[idx][1].axis("off")
        axs[idx][2].imshow(nucleus_mask, cmap='gray')
        axs[idx][2].set_title("Mask: nucleus")
        axs[idx][2].axis("off")
    plt.tight_layout()
    plt.show()



def plot_images_from_filenames(model, device, folder_path, filenames, 
                             threshold_nuclei=0.2, threshold_cell=0.5, 
                             select_nucleus=False, 
                             apply_morphology=False, 
                             kernel_size=5):         
    """
    Rysuje obrazy i ich maski, opcjonalnie stosując operacje morfologiczne
    (otwieranie i zamykanie) w celu wygładzenia masek.
    
    Args:
        apply_morphology (bool): Czy zastosować operacje morfologiczne.
        kernel_size (int): Rozmiar kernela, jeśli apply_morphology=True.
    """
    n = len(filenames)
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axs = [axs]
        
    for idx, filename in enumerate(filenames):
        image_path = os.path.join(folder_path, filename)
        pil_image, input_tensor = preprocess_image(image_path)
        
        predicted_masks = predict_masks(model, input_tensor, device, threshold_nuclei, threshold_cell)
        cell_mask = predicted_masks[0]
        nucleus_mask = predicted_masks[1]
        
        if select_nucleus:
            nucleus_mask_try = select_best_nucleus(nucleus_mask, pil_image.size[::-1])
            if nucleus_mask_try.any():
                nucleus_mask = nucleus_mask_try

        cell_mask_final = cell_mask
        nucleus_mask_final = nucleus_mask
        
        if apply_morphology and kernel_size > 0:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            cell_mask_uint8 = (cell_mask.astype(np.uint8) * 255)
            nucleus_mask_uint8 = (nucleus_mask.astype(np.uint8) * 255)

            cell_mask_opened = cv2.morphologyEx(cell_mask_uint8, cv2.MORPH_OPEN, kernel)
            nucleus_mask_opened = cv2.morphologyEx(nucleus_mask_uint8, cv2.MORPH_OPEN, kernel)

            cell_mask_final = cv2.morphologyEx(cell_mask_opened, cv2.MORPH_CLOSE, kernel)
            nucleus_mask_final = cv2.morphologyEx(nucleus_mask_opened, cv2.MORPH_CLOSE, kernel)
        

        axs[idx][0].imshow(pil_image)
        axs[idx][0].set_title(f"Oryginal: {filename}")
        axs[idx][0].axis("off")

        axs[idx][1].imshow(cell_mask_final, cmap='gray')
        axs[idx][1].set_title("Mask: cell")
        axs[idx][1].axis("off")

        axs[idx][2].imshow(nucleus_mask_final, cmap='gray')
        axs[idx][2].set_title("Mask: nucleus")
        axs[idx][2].axis("off")

    plt.tight_layout()
    plt.show()