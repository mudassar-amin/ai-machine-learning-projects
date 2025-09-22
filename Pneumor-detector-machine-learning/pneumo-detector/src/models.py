from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models

def create_model(name: str = "densenet121", pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        in_feats = m.classifier.in_features
        m.classifier = nn.Linear(in_feats, 1)
        return m
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, 1)
        return m
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, 1)
        return m
    else:
        raise ValueError(f"Unknown model name: {name}")