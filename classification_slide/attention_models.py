from torch import softmax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os



class AttentionMIL(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_classes=3, dropout=0.3):
        super(AttentionMIL, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        num_cells = x.shape[1]
        
        x_flat = x.view(-1, x.shape[2])
        features = self.feature_extractor(x_flat)
        features = features.view(batch_size, num_cells, -1)
        
        attention_scores = self.attention(features)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        slide_features = torch.sum(attention_weights * features, dim=1)
        
        output = self.classifier(slide_features)
        
        return output, attention_weights
    

def collate_fn(batch):
    slides, labels, filenames = zip(*batch)
    max_cells = max([s.shape[0] for s in slides])
    padded_slides = []
    for slide in slides:
        num_cells = slide.shape[0]
        if num_cells < max_cells:
            padding = torch.zeros(max_cells - num_cells, slide.shape[1])
            padded_slide = torch.cat([slide, padding], dim=0)
        else:
            padded_slide = slide
        padded_slides.append(padded_slide)
    
    return torch.stack(padded_slides), torch.stack(labels), filenames

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for slides, labels, _ in dataloader:
        slides, labels = slides.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(slides)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), 100 * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for slides, labels, _ in dataloader:
            slides, labels = slides.to(device), labels.to(device)
            
            outputs, _ = model(slides)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), 100 * correct / total

def evaluate_detailed(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for slides, labels, filenames in dataloader:
            slides, labels = slides.to(device), labels.to(device)
            outputs, _ = model(slides)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    
    print("\n📊 Confusion Matrix:")
    print(cm)
    print("\n📈 Classification Report:")
    print(report)
    
    wrong_indices = np.where(all_preds != all_labels)[0]
    if len(wrong_indices) > 0:
        print(f"\n❌ Błędnie sklasyfikowane ({len(wrong_indices)} próbek):")
        for idx in wrong_indices[:10]: 
            print(f"  {all_filenames[idx]}: True={class_names[all_labels[idx]]}, Pred={class_names[all_preds[idx]]}")
    
    return all_preds, all_labels, all_filenames

def predict_attention(model, slide, filename, device, class_names, visualize=False):
    model.eval()
    with torch.no_grad():
        slide_tensor = torch.FloatTensor(slide).unsqueeze(0).to(device)
        output, attention_weights = model(slide_tensor)
        
        _, predicted = torch.max(output, 1)
        attn = attention_weights.squeeze().cpu().numpy()

        probs = softmax(output, dim=1)
        probs = probs.squeeze(0).cpu().numpy()
        
        if visualize:
            print(f"\n🔍 Analiza slajdu: {filename}")
            print(f"  Predicted class: {class_names[predicted.item()]}")
            print(f"  Top 5 important cells (attention weights):")
            top_indices = np.argsort(attn.flatten())[-5:][::-1]
            for i, idx in enumerate(top_indices):
                if idx < len(slide):
                    print(f"    Cell {idx}: weight={attn[idx]:.4f}, probs={slide[idx]}")
        
        return predicted.item(), attn, probs