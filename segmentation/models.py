import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# --- U-Net Model ---
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

        self.bottleneck = DoubleConv(128, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.output_layer = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
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

        # logowanie do TensorBoard
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

def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, log_dir="runs/exp1"):
    model.to(device)
    writer = SummaryWriter(log_dir)
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)

            # Loguj przykładowe obrazy co 100 kroków
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

        # Ewaluacja na zbiorze walidacyjnym
        val_loss = evaluate(model, val_loader, loss_fn, device, writer, global_step)
        print(f"Epoch {epoch+1} Val loss: {val_loss:.4f}")

    writer.close()