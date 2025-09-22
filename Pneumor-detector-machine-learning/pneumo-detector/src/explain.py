from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_target_layers(model):
    name = model.__class__.__name__.lower()
    # Handle ResNet and DenseNet common last conv blocks
    if "resnet" in name:
        return [model.layer4[-1]]
    if "densenet" in name:
        return [model.features.denseblock4]
    # Fallback: try to find a last features attr
    if hasattr(model, "features"):
        return [model.features]
    raise ValueError("Could not infer target layer for Grad-CAM.")

def load_image(image_path: str, img_size: int):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((img_size, img_size))
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(img_resized).unsqueeze(0)
    # normalize like ImageNet
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = norm(input_tensor)
    rgb = np.array(img_resized).astype(np.float32)/255.0
    return input_tensor, rgb

@torch.no_grad()
def gradcam_overlay(model, image_path: str, img_size: int, out_path: str, target_class: int = 1):
    device = next(model.parameters()).device
    input_tensor, rgb = load_image(image_path, img_size)
    input_tensor = input_tensor.to(device)

    targets = [ClassifierOutputTarget(target_class)]
    target_layers = get_target_layers(model)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device.type=="cuda")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))