
import argparse, os, sys, zipfile, shutil, json, random
from pathlib import Path
import torch.nn.functional as F
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_extracted(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Unzipping {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    cx = out_dir / "chest_xray"
    if not cx.exists():
        cand = list(out_dir.glob("**/chest_xray"))
        if cand:
            cx = cand[0]
        else:
            raise FileNotFoundError("Could not find 'chest_xray' folder inside the zip contents.")
    final = out_dir / "chest_xray"
    if cx.resolve() != final.resolve():
        if final.exists(): shutil.rmtree(final)
        shutil.move(str(cx), str(final))
    return final

def verify_structure(cx_root: Path):
    required = [cx_root/sp/cls for sp in ["train","val","test"] for cls in ["NORMAL","PNEUMONIA"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing folders:\n" + "\n".join(missing))

def make_transforms(img_size: int, split: str):
    # Use pure-torch conversions (no NumPy)
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.PILToTensor(),                       # -> torch.uint8 [0,255]
            transforms.ConvertImageDtype(torch.float32),    # -> float32 [0,1]
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])



def create_loaders(cx_root: Path, img_size: int, batch_size: int, num_workers: int):
    train_ds = datasets.ImageFolder(cx_root/"train", transform=make_transforms(img_size, "train"))
    val_ds   = datasets.ImageFolder(cx_root/"val",   transform=make_transforms(img_size, "val"))
    test_ds  = datasets.ImageFolder(cx_root/"test",  transform=make_transforms(img_size, "test"))
    classes = train_ds.classes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, classes, train_ds, val_ds, test_ds

def create_resnet18(pretrained=True) -> nn.Module:
    try:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    except Exception:
        print("⚠️ Could not load pretrained weights; using random init.")
        m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m

def collect_logits(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = model(x).squeeze(1)
            logits_list.append(z.detach().cpu().numpy())
            labels_list.append(y.detach().cpu().numpy())
    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)
    probs = 1/(1+np.exp(-logits))
    return logits, probs, labels

def pick_threshold_for_sensitivity(labels, probs, target=0.90):
    fpr, tpr, thr = roc_curve(labels, probs)
    mask = tpr >= target
    if not mask.any():
        idx = int(np.argmax(tpr))
    else:
        idx = int(np.arange(len(thr))[mask][np.argmin(fpr[mask])])
    return float(thr[idx])

def plot_roc_pr(labels, probs, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC - {tag}")
    plt.legend(loc="lower right"); plt.savefig(out_dir / f"roc_{tag}.png", dpi=150, bbox_inches="tight"); plt.close()
    prec, rec, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    plt.figure(); plt.plot(rec, prec, label=f"AUPRC={auprc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {tag}")
    plt.legend(loc="lower left"); plt.savefig(out_dir / f"pr_{tag}.png", dpi=150, bbox_inches="tight"); plt.close()

def save_confusion_and_report(labels, probs, thr, out_dir: Path, tag: str, class_names):
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0,1])
    plt.figure(); plt.imshow(cm)
    plt.title(f"Confusion - {tag}")
    plt.xticks([0,1], class_names); plt.yticks([0,1], class_names)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
    plt.savefig(out_dir / f"confusion_{tag}.png", dpi=150, bbox_inches="tight"); plt.close()
    rep = classification_report(labels, preds, target_names=class_names, digits=4)
    with open(out_dir / f"classification_report_{tag}.txt", "w") as f: f.write(rep)
    return cm, rep

# ---- Minimal built-in Grad-CAM (no external package) ----
def gradcam_resnet18(model: nn.Module, x: torch.Tensor, device: torch.device, target_class: int = 1):
    """
    x: (1,3,H,W) normalized tensor
    Returns a CAM resized to (H,W) in [0,1].
    """
    model.eval()
    # last conv in layer4
    target_layer = model.layer4[-1].conv2 if hasattr(model.layer4[-1], "conv2") else model.layer4[-1]
    activations, gradients = [], []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(module, grad_input, grad_output):
        # grad_output[0] is dL/dA with shape (N,C,H,W)
        gradients.append(grad_output[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x = x.to(device)
    H_in, W_in = x.shape[-2:]
    model.zero_grad()
    logits = model(x).squeeze(1)      # (N,)
    score = logits[0]                 # positive-class logit
    score.backward()

    h1.remove(); h2.remove()

    A = activations[-1][0]            # (C,h,w)
    G = gradients[-1][0]              # (C,h,w)

    weights = G.mean(dim=(1, 2))      # (C,)
    cam = torch.relu((weights[:, None, None] * A).sum(dim=0))  # (h,w)

    # Normalize then upsample to input size
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cam.unsqueeze(0).unsqueeze(0)                # (1,1,h,w)
    cam = F.interpolate(cam, size=(H_in, W_in), mode="bilinear", align_corners=False)
    cam = cam.squeeze(0).squeeze(0)                    # (H_in, W_in)
    return cam.detach().cpu().numpy()


def overlay_cam(rgb_np: np.ndarray, cam: np.ndarray, alpha: float = 0.35):
    """
    rgb_np: (H,W,3) float in [0,1]
    cam:    (H,W)    float in [0,1]
    """
    cmap = matplotlib.colormaps.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]  # (H,W,3)
    out = (1.0 - alpha) * rgb_np + alpha * heatmap
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


# ---- Training ----
def train_main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and args.amp

    if args.zip is not None:
        zip_path = Path(args.zip)
        if not zip_path.exists(): raise FileNotFoundError(f"ZIP not found: {zip_path}")
        cx_root = ensure_extracted(zip_path, Path(args.data_dir))
    else:
        if args.data_root is None: raise ValueError("Provide either --zip or --data-root")
        cx_root = Path(args.data_root)
    verify_structure(cx_root)
    print(f"✅ Dataset at: {cx_root}")

    train_loader, val_loader, test_loader, classes, *_ = create_loaders(
        cx_root, args.img_size, args.batch_size, args.num_workers
    )

    model = create_resnet18(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_ckpt = out_dir / "best.ckpt"

    for epoch in range(1, args.epochs+1):
        model.train(); running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if amp:
                with torch.cuda.amp.autocast():
                    logits = model(x).squeeze(1)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                logits = model(x).squeeze(1); loss = criterion(logits, y)
                loss.backward(); optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        _, v_probs, v_labels = collect_logits(model, val_loader, device)
        v_auroc = roc_auc_score(v_labels, v_probs)
        if v_auroc > best_val:
            best_val = v_auroc
            torch.save({"model": model.state_dict(),
                        "meta": {"img_size": args.img_size, "classes": classes}}, best_ckpt)
        print(f"train_loss={train_loss:.4f}  val_AUROC={v_auroc:.4f}  (best={best_val:.4f})")

    ck = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck["model"])

    _, v_probs, v_labels = collect_logits(model, val_loader, device)
    thr = pick_threshold_for_sensitivity(v_labels, v_probs, target=args.target_sensitivity)

    plot_roc_pr(v_labels, v_probs, out_dir, "val")
    save_confusion_and_report(v_labels, v_probs, thr, out_dir, "val", classes)

    _, t_probs, t_labels = collect_logits(model, test_loader, device)
    plot_roc_pr(t_labels, t_probs, out_dir, "test")
    save_confusion_and_report(t_labels, t_probs, thr, out_dir, "test", classes)

    metrics = {
        "val": {"AUROC": float(roc_auc_score(v_labels, v_probs)),
                "AUPRC": float(average_precision_score(v_labels, v_probs)),
                "threshold": float(thr)},
        "test": {"AUROC": float(roc_auc_score(t_labels, t_probs)),
                 "AUPRC": float(average_precision_score(t_labels, t_probs)),
                 "threshold": float(thr)}
    }
    with open(out_dir / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    print("\n=== Metrics ==="); print(json.dumps(metrics, indent=2))
    print(f"\nArtifacts saved in: {out_dir}")
    print(f"Best checkpoint: {best_ckpt}")

# ---- Inference + built-in Grad-CAM ----
# ---- Inference + built-in Grad-CAM ----
def infer_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    meta = ck.get("meta", {})
    img_size = int(meta.get("img_size", 320))

    model = create_resnet18(pretrained=True)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    # load + resize once
    img = Image.open(args.image).convert("RGB").resize((img_size, img_size))

    # PURE-TORCH transforms (no NumPy path)
    t = transforms.Compose([
        transforms.PILToTensor(),                    # uint8 [0,255] -> tensor
        transforms.ConvertImageDtype(torch.float32), # float32 [0,1]
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    x = t(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze(1)).item()
    print(f"Predicted probability of Pneumonia: {prob:.4f}")

    if args.overlay:
        # only for saving the visualization we make an RGB array
        rgb = np.array(img).astype(np.float32) / 255.0
        cam = gradcam_resnet18(model, x, device, target_class=1)
        overlay = overlay_cam(rgb, cam, alpha=0.35)
        Image.fromarray(overlay).save(args.overlay)
        print(f"Grad-CAM overlay saved to {args.overlay}")


def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection (single file, built-in Grad-CAM)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train and evaluate the model")
    p_train.add_argument("--zip", type=str, default=None, help="Path to local chest-xray-pneumonia.zip")
    p_train.add_argument("--data-root", type=str, default=None, help="Path to existing chest_xray folder")
    p_train.add_argument("--data-dir", type=str, default="data", help="Where to extract ZIP (default: ./data)")
    p_train.add_argument("--out-dir", type=str, default="outputs/resnet18_baseline", help="Where to save outputs")
    p_train.add_argument("--img-size", type=int, default=320)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--pos-weight", type=float, default=1.5)
    p_train.add_argument("--amp", action="store_true", help="Enable mixed precision (GPU only)")
    p_train.add_argument("--target-sensitivity", type=float, default=0.90)
    p_train.set_defaults(func=train_main)

    p_infer = sub.add_parser("infer", help="Run single-image inference (+ Grad-CAM overlay)")
    p_infer.add_argument("--ckpt", type=str, required=True, help="Path to best.ckpt")
    p_infer.add_argument("--image", type=str, required=True, help="Path to input chest X-ray image")
    p_infer.add_argument("--overlay", type=str, default=None, help="If set, save Grad-CAM overlay here (PNG)")
    p_infer.set_defaults(func=infer_main)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()