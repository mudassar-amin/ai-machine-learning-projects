import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .models import create_model
from .explain import gradcam_overlay

def load_ckpt(ckpt_path, device):
    import yaml
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]
    model = create_model(cfg["model"]["name"], cfg["model"].get("pretrained", True))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--overlay", type=str, default=None, help="If set, write Grad-CAM overlay to this path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_ckpt(args.ckpt, device)

    img_size = cfg["data"]["img_size"]
    img = Image.open(args.image).convert("RGB").resize((img_size, img_size))
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits).item()

    print(f"Predicted probability of Pneumonia: {prob:.4f}")
    if args.overlay:
        gradcam_overlay(model, args.image, img_size, args.overlay, target_class=1)
        print(f"Grad-CAM overlay saved to {args.overlay}")

if __name__ == "__main__":
    main()