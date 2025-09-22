import os
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_transforms(img_size: int, split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def get_loaders(root: str, img_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Expected {train_dir} with NORMAL/ and PNEUMONIA/ subfolders.")
    if not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected {val_dir} and {test_dir} with NORMAL/ and PNEUMONIA/ subfolders.")

    train_ds = datasets.ImageFolder(train_dir, transform=make_transforms(img_size, "train"))
    val_ds   = datasets.ImageFolder(val_dir,   transform=make_transforms(img_size, "val"))
    test_ds  = datasets.ImageFolder(test_dir,  transform=make_transforms(img_size, "test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes