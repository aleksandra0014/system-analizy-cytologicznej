import os
import torch
from torchvision import transforms
from PIL import Image
from models import CytologyClassifier
import matplotlib.pyplot as plt
import numpy as np
import cv2


def grad_cam(architecture, model_path, image_path):
    class_names = {0: "HSIL", 1: "LSIL", 2: "NSIL"}
    model = CytologyClassifier(architecture=architecture, num_classes=len(class_names))
    model.load(model_path)
    model.model.eval()

    # Hooki: aktywacje + gradienty
    activations = {}
    gradients = {}
    def forward_hook(module, input, output):
                activations["conv_final"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
            gradients["conv_final"] = grad_output[0].detach()

    if architecture == 'vgg16':
        target_layer = model.model.features[29]
    elif architecture == 'resnet18':
        target_layer = model.model.layer4
    elif architecture == 'custom_cnn':
        target_layer = model.model.features[-3]  

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    output = model.model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()

    model.model.zero_grad()
    output[0, class_idx].backward()

    # === 1. ORYGINAŁ ===
    original = image.resize((224, 224))
    original_np = np.array(original)

    # === 2. AKTYWACJE ===
    feature_map = activations["conv_final"].squeeze(0)      
    # elif architecture == 'resnet18':
    #     feature_map = activations["layer4"].squeeze(0)          
    avg_activation = torch.mean(feature_map, dim=0).cpu().numpy()
    avg_activation -= avg_activation.min()
    avg_activation /= avg_activation.max()
    avg_activation_img = np.uint8(255 * avg_activation)
    avg_activation_img = cv2.resize(avg_activation_img, (224, 224))
    avg_activation_color = cv2.applyColorMap(avg_activation_img, cv2.COLORMAP_JET)

    # === 3. GRAD-CAM ===
    # if architecture == 'vgg16':
    grads = gradients["conv_final"].squeeze(0)              
    # elif architecture == 'resnet18':
    # grads = gradients["layer4"].squeeze(0)                 
    weights = torch.mean(grads, dim=(1, 2))                
    cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * feature_map[i]
    cam = torch.clamp(cam, min=0).cpu().numpy()
    cam -= cam.min()
    cam /= cam.max()
    cam_img = np.uint8(255 * cam)
    cam_img = cv2.resize(cam_img, (224, 224))
    gradcam_color = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, gradcam_color, 0.4, 0)
    
    # === WYKRES 3-OBRAZKOWY ===
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_np)
    plt.title("Oryginalne zdjęcie")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(avg_activation_color, cv2.COLOR_BGR2RGB))
    plt.title("Aktywacja (średnia z conv)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)) 
    plt.title(f"Grad-CAM nałożony ({class_names.get(class_idx, class_idx)})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    architecture = 'resnet18' 
    model_path = r'C:\Users\aleks\OneDrive\Documents\inzynierka\classification\classification_models\resnet18\32_0_0001_50_0608.pth'
    for subfolder in [
        r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\TEST\HSIL',
        r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\TEST\LSIL',
        r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\TEST\NSIL'
    ]:
        for filename in os.listdir(subfolder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(subfolder, filename)
                print(f"Processing: {image_path}")
            grad_cam(architecture, model_path, image_path)