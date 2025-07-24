import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from classification.models import CNNClassifier, CytologyClassifier 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from typing import List, Tuple, Optional, Union

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str, model_name: str = 'resnet'):
        """
        GradCAM implementation dla CNNClassifier, ResNet, VGG
        
        Args:
            model: Wytrenowany model
            target_layer: Nazwa warstwy docelowej (np. 'features.21' dla VGG, 'layer4.1' dla ResNet)
            model_name: 'resnet', 'vgg', 'custom_cnn'
        """
        self.model = model
        self.target_layer = target_layer
        self.model_name = model_name
        self.gradients = None
        self.activations = None
        
        # Rejestracja hooków
        self._register_hooks()
        
    def _register_hooks(self):
        """Rejestruje hooki do przechwytywania gradientów i aktywacji"""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Znajdź warstwę docelową
        target_module = self._get_target_layer()
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
        
    def load_and_preprocess_image(self, image_path: str, transform: Optional[transforms.Compose] = None) -> torch.Tensor:
        """
        Ładuje i preprocessuje obraz z pliku
        
        Args:
            image_path: Ścieżka do pliku obrazu
            transform: Opcjonalne transformacje (jeśli None, używa domyślnych)
            
        Returns:
            Preprocessowany tensor obrazu (1, 3, H, W)
        """
        if transform is None:
            # Domyślne transformacje dla obrazów 128x128
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
        # Załaduj obraz
        image = Image.open(image_path).convert('RGB')
        
        # Zastosuj transformacje i dodaj batch dimension
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor
        
    def _get_target_layer(self):
        """Zwraca referencję do warstwy docelowej, obsługuje VGG, ResNet, custom_cnn"""
        parts = self.target_layer.split('.')
        module = self.model
        
        if hasattr(self.model, 'model'):
            module = self.model.model  # CytologyClassifier/CNNClassifier wrapper
        if self.model_name == 'vgg':
            # VGG: warstwy w model.features
            if parts[0] == 'features':
                module = module.features
                for part in parts[1:]:
                    module = module[int(part)]
            else:
                module = getattr(module, parts[0])
                for part in parts[1:]:
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
        elif self.model_name == 'resnet':
            # ResNet: warstwy typu 'layer4.1', 'layer3.0', 'conv1', ...
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
        elif self.model_name == 'custom_cnn':
            # custom_cnn: warstwy w model.features
            if parts[0] == 'features':
                module = module.features
                for part in parts[1:]:
                    module = module[int(part)]
            else:
                module = getattr(module, parts[0])
                for part in parts[1:]:
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
        else:
            # fallback
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
                
        return module
        
    def generate_cam(self, input_data: Union[torch.Tensor, str], 
                    class_idx: Optional[int] = None, 
                    transform: Optional[transforms.Compose] = None) -> Tuple[np.ndarray, int]:
        """
        Generuje mapę aktywacji dla danego obrazu
        
        Args:
            input_data: Tensor wejściowy (1, 3, H, W) lub ścieżka do pliku obrazu
            class_idx: Indeks klasy (jeśli None, używa przewidywanej klasy)
            transform: Transformacje dla obrazu (tylko gdy input_data to ścieżka)
            
        Returns:
            Tuple: (Mapa CAM jako numpy array, indeks klasy)
        """
        # Obsłuż różne typy wejścia
        if isinstance(input_data, str):
            input_tensor = self.load_and_preprocess_image(input_data, transform)
        else:
            input_tensor = input_data
            
        self.model.eval()
        
        # Forward pass
        input_tensor.requires_grad_()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Oblicz wagi z gradientów
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global Average Pooling gradientów
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalizacja
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
        
    def visualize(self, input_data: Union[torch.Tensor, str], 
                  class_idx: Optional[int] = None, 
                  alpha: float = 0.4, 
                  colormap: str = 'jet',
                  transform: Optional[transforms.Compose] = None) -> Tuple[np.ndarray, int]:
        """
        Tworzy wizualizację GradCAM
        
        Args:
            input_data: Tensor wejściowy (1, 3, H, W) lub ścieżka do pliku obrazu
            class_idx: Indeks klasy
            alpha: Przezroczystość overlay
            colormap: Mapa kolorów
            transform: Transformacje dla obrazu (tylko gdy input_data to ścieżka)
            
        Returns:
            Tuple: (Obraz z nałożoną mapą CAM, indeks klasy)
        """
        # Obsłuż różne typy wejścia
        if isinstance(input_data, str):
            input_tensor = self.load_and_preprocess_image(input_data, transform)
        else:
            input_tensor = input_data
            
        # Generuj CAM
        cam, predicted_class = self.generate_cam(input_tensor, class_idx)
        
        # Przygotuj oryginalny obraz
        original_img = input_tensor.squeeze().cpu().detach().numpy()
        original_img = np.transpose(original_img, (1, 2, 0))
        
        # Denormalizacja (zakładając normalizację ImageNet)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # original_img = original_img * std + mean
        # original_img = np.clip(original_img, 0, 1)
        
        # Przeskaluj CAM do rozmiaru obrazu
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # Zastosuj mapę kolorów
        heatmap = plt.cm.get_cmap(colormap)(cam_resized)
        heatmap = heatmap[:, :, :3]  # Usuń kanał alpha
        
        # Nałóż heatmap na oryginalny obraz
        superimposed_img = heatmap * alpha + original_img * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 1)
        
        return superimposed_img, predicted_class

def plot_gradcam_results(model, input_data: Union[torch.Tensor, str], 
                        class_names=None, target_layers=None, 
                        transform: Optional[transforms.Compose] = None, model_name: str = 'resnet'):
    """
    Funkcja pomocnicza do wizualizacji wyników GradCAM dla różnych warstw
    
    Args:
        model: Wytrenowany model
        input_data: Tensor obrazu (1, 3, H, W) lub ścieżka do pliku obrazu
        class_names: Lista nazw klas
        target_layers: Lista warstwy do analizy
        transform: Transformacje dla obrazu (tylko gdy input_data to ścieżka)
        model_name: 'resnet', 'vgg', 'custom_cnn'
    """
    if target_layers is None:
        if model_name == 'vgg':
            target_layers = ['features.5', 'features.11', 'features.17', 'features.23']
        elif model_name == 'resnet':
            target_layers = ['layer1.0', 'layer2.0', 'layer3.1', 'layer4.1']
        else:
            target_layers = ['features.21']
    
    # Obsłuż różne typy wejścia
    if isinstance(input_data, str):
        # Załaduj obraz do wyświetlenia oryginalnego
        if transform is None:
            display_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
        else:
            display_transform = transform
        original_image = Image.open(input_data).convert('RGB')
        display_tensor = display_transform(original_image).unsqueeze(0)
    else:
        display_tensor = input_data
    
    fig, axes = plt.subplots(2, len(target_layers), figsize=(4*len(target_layers), 8))
    if len(target_layers) == 1:
        axes = axes.reshape(2, 1)
    
    for i, layer in enumerate(target_layers):
        # Utwórz GradCAM dla tej warstwy
        gradcam = GradCAM(model, layer, model_name=model_name)
        
        # Generuj wizualizację
        cam_img, pred_class = gradcam.visualize(input_data, transform=transform)
        
        # Generuj samą mapę CAM
        cam_only, _ = gradcam.generate_cam(input_data, transform=transform)
        
        # Wyświetl nałożony obraz
        axes[0, i].imshow(cam_img)
        axes[0, i].set_title(f'GradCAM: {layer}')
        axes[0, i].axis('off')
        
        # Wyświetl samą mapę CAM
        axes[1, i].imshow(cam_only, cmap='jet')
        axes[1, i].set_title(f'CAM tylko: {layer}')
        axes[1, i].axis('off')
    
    # Dodaj informację o przewidywanej klasie
    if class_names and pred_class < len(class_names):
        fig.suptitle(f'Przewidywana klasa: {class_names[pred_class]} (indeks: {pred_class})', 
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Przewidywana klasa: {pred_class}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_gradcam_results2(model, input_data: Union[torch.Tensor, str], 
                         class_names=None, target_layers=None, 
                         transform: Optional[transforms.Compose] = None, model_name: str = 'resnet') -> plt.Figure:
    """
    Zwraca obiekt matplotlib.Figure z wynikami GradCAM dla Streamlit.
    model_name: 'resnet', 'vgg', 'custom_cnn'
    """
    import matplotlib.pyplot as plt
    if target_layers is None:
        if model_name == 'vgg' or model_name == 'vgg16':
            target_layers = ['features.5', 'features.11', 'features.17', 'features.23']
        elif model_name == 'resnet' or model_name == 'resnet18':
            target_layers = ['layer1.0', 'layer2.0', 'layer3.1', 'layer4.1']
        else:
            target_layers = ['features.21']
    if isinstance(input_data, str):
        if transform is None:
            display_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            display_transform = transform
        original_image = Image.open(input_data).convert('RGB')
        display_tensor = display_transform(original_image).unsqueeze(0)
    else:
        display_tensor = input_data
    fig, axes = plt.subplots(2, len(target_layers), figsize=(4 * len(target_layers), 8))
    if len(target_layers) == 1:
        axes = axes.reshape(2, 1)
    for i, layer in enumerate(target_layers):
        gradcam = GradCAM(model, layer, model_name=model_name)
        cam_img, pred_class = gradcam.visualize(input_data, transform=transform)
        cam_only, _ = gradcam.generate_cam(input_data, transform=transform)
        axes[0, i].imshow(cam_img)
        axes[0, i].set_title(f'GradCAM: {layer}')
        axes[0, i].axis('off')
        axes[1, i].imshow(cam_only, cmap='jet')
        axes[1, i].set_title(f'CAM tylko: {layer}')
        axes[1, i].axis('off')
    if class_names and pred_class < len(class_names):
        fig.suptitle(f'Przewidywana klasa: {class_names[pred_class]} (indeks: {pred_class})', 
                     fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Przewidywana klasa: {pred_class}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def analyze_image_from_path(model, image_path: str, target_layer: str = 'features.23', 
                           class_names: Optional[List[str]] = None,
                           custom_transform: Optional[transforms.Compose] = None):
    """
    Funkcja pomocnicza do szybkiej analizy obrazu z pliku
    
    Args:
        model: Wytrenowany model
        image_path: Ścieżka do pliku obrazu
        target_layer: Warstwa do analizy
        class_names: Lista nazw klas
        custom_transform: Opcjonalne niestandardowe transformacje
    """
    # Utwórz GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Analizuj obraz
    cam_image, predicted_class = gradcam.visualize(image_path, transform=custom_transform)
    
    # Załaduj oryginalny obraz do wyświetlenia
    original_image = Image.open(image_path).convert('RGB')
    
    # Wyświetl wyniki
    plt.figure(figsize=(15, 5))
    
    # Oryginalny obraz
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Oryginalny obraz')
    plt.axis('off')
    
    # GradCAM overlay
    plt.subplot(1, 3, 2)
    plt.imshow(cam_image)
    if class_names and predicted_class < len(class_names):
        plt.title(f'GradCAM\nKlasa: {class_names[predicted_class]}')
    else:
        plt.title(f'GradCAM\nKlasa: {predicted_class}')
    plt.axis('off')
    
    # Sama mapa CAM
    plt.subplot(1, 3, 3)
    cam_only, _ = gradcam.generate_cam(image_path, transform=custom_transform)
    plt.imshow(cam_only, cmap='jet')
    plt.title('Mapa aktywacji')
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class

if __name__=='__main__':
    model = CNNClassifier(num_classes=3)  
    model.load_state_dict(torch.load(r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\mine_cnn_0.001.32.50_v2.pth'))
    model.eval()

    image_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\NSIL\61b_1.bmp"
    for image_path in [r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\HSIL\6b_2.bmp', r'data\data_single\LSIL\29b_1.bmp', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\LSIL\3b_1.bmp', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\LSIL\33b_1.bmp', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\LSIL\9b_2.bmp', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\HSIL\19b_1.bmp', r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\NSIL\16b_1.bmp']:
        predicted_class = analyze_image_from_path(
            model, 
            image_path, 
            target_layer='features.23',
            class_names=['HSIL', 'LSIL', 'NSIL']
        )

        plot_gradcam_results(
            model, 
            image_path, 
            class_names=['HSIL', 'LSIL', 'NSIL'],
            model_name='custom_cnn',
        )


