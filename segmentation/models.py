from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

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


def test_model(model, test_loader, device):
    model.eval()
    model.to(device)

    dice_all = []
    iou_all = []
    precision_all = []
    recall_all = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = torch.sigmoid(model(images))
            preds = (outputs > 0.5).float()

            for cls in range(preds.shape[1]):  
                dice_all.append(dice_score(preds[:, cls], masks[:, cls]))
                iou_all.append(iou_score(preds[:, cls], masks[:, cls]))
                precision_all.append(precision(preds[:, cls], masks[:, cls]))
                recall_all.append(recall(preds[:, cls], masks[:, cls]))

    def mean_metric(metric_list):
        all_vals = torch.cat(metric_list, dim=0)
        return all_vals.mean().item()

    print("\n📊 Results:")
    print(f"Dice:     {mean_metric(dice_all):.4f}")
    print(f"IoU:      {mean_metric(iou_all):.4f}")
    print(f"Precision:{mean_metric(precision_all):.4f}")
    print(f"Recall:   {mean_metric(recall_all):.4f}")

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

def predict_masks(model, image_tensor, device, threshold_nuclei=0.3, threshold_cell=0.5):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        preds = torch.sigmoid(output).squeeze(0)  # [2, H, W]

        mask_cell = (preds[0] > threshold_cell).float().cpu().numpy()  # jądro
        mask_nucleus = (preds[1] > threshold_nuclei).float().cpu().numpy()      # komórka

    return [mask_cell, mask_nucleus]  

def plot_results(model, image_path, device,  threshold_nuclei=0.3, threshold_cell=0.5):
    pil_image, input_tensor = preprocess_image(image_path)
    predicted_masks = predict_masks(model, input_tensor, device, threshold_nuclei, threshold_cell)
    cell_mask = predicted_masks[0]
    nucleus_mask = predicted_masks[1]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(pil_image)
    axs[0].set_title("Oryginal")

    axs[1].imshow(cell_mask.numpy(), cmap='gray')
    axs[1].set_title("Mask: cell")

    axs[2].imshow(nucleus_mask.numpy(), cmap='gray')
    axs[2].set_title("Mask: nucleus")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



