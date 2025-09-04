import pathlib
from typing import Optional
import torch
from classification.models import CytologyClassifier
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def gradcam_on_image(model_wrapper: CytologyClassifier, architecture: str, img_path: pathlib.Path, device: Optional[torch.device] = None):
    model = model_wrapper.model
    model.eval()
    device = device or next(model.parameters()).device

    if architecture == 'vgg16':
        target_layer: nn.Module = model.features[28] 
    elif architecture == 'resnet18':
        target_layer = model.layer4[-1].conv2
    else:
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise RuntimeError("No Conv2d layer found for Grad-CAM")
        target_layer = convs[-1]

    activations = {}

    def forward_hook(_m, _inp, out):
        activations["x"] = out
        out.retain_grad()  
    fh = target_layer.register_forward_hook(forward_hook)

    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        pil_img = Image.open(str(img_path)).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.set_grad_enabled(True):
            output = model(input_tensor)
            class_idx = int(torch.argmax(output, dim=1).item())
            model.zero_grad(set_to_none=True)
            output[0, class_idx].backward()

        feat = activations["x"].detach()            # [1,C,H,W]
        grads = activations["x"].grad.detach()      # [1,C,H,W]

        weights = grads.mean(dim=(2, 3), keepdim=True)   # [1,C,1,1]
        cam = (weights * feat).sum(dim=1).squeeze(0)     # [H,W]
        cam = torch.relu(cam).cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() or 1.0)

        cam_img = (cam * 255.0).astype("uint8")
        cam_img = cv2.resize(cam_img, (224, 224))
        cam_color = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

        original_np = np.array(pil_img.resize((224, 224)))
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.6, cam_color, 0.4, 0)

        avg_act = feat.mean(dim=1).squeeze(0).cpu().numpy()
        avg_act -= avg_act.min()
        avg_act /= (avg_act.max() or 1.0)
        avg_img = (avg_act * 255.0).astype("uint8")
        avg_img = cv2.resize(avg_img, (224, 224))
        avg_color = cv2.applyColorMap(avg_img, cv2.COLORMAP_JET)

        return overlay, cam_color, avg_color, class_idx
    finally:
        try:
            fh.remove()
        except Exception:
            pass