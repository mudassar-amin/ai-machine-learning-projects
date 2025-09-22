import argparse, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

from .datasets import get_loaders
from .models import create_model
from .evaluate import evaluate_split, collect_logits, pick_threshold_for_sensitivity

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bce_loss(pos_weight):
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float))

def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["eval"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader, classes = get_loaders(
        cfg["data"]["root"],
        cfg["data"]["img_size"],
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
    )

    # Model
    model = create_model(cfg["model"]["name"], cfg["model"].get("pretrained", True)).to(device)

    # Optim & loss
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    criterion = bce_loss(cfg["training"]["pos_weight"]).to(device)
    scaler = GradScaler(enabled=bool(cfg["training"]["amp"]))

    save_dir = Path(cfg["eval"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_val_auroc = -1.0

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
        # Evaluate on val
        _, val_probs, val_labels = collect_logits(model, val_loader, device)
        from sklearn.metrics import roc_auc_score
        val_auroc = roc_auc_score(val_labels, val_probs)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({"model": model.state_dict(), "cfg": cfg}, save_dir / "best.ckpt")
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | train_loss={train_loss:.4f} | val_AUROC={val_auroc:.4f}")

    # Load best and finalize eval
    ckpt = torch.load(save_dir / "best.ckpt", map_location=device)
    model.load_state_dict(ckpt["model"])

    # Choose threshold on val to hit target sensitivity
    _, val_probs, val_labels = collect_logits(model, val_loader, device)
    thr = pick_threshold_for_sensitivity(val_labels, val_probs, target_sens=cfg["eval"]["target_sensitivity"])

    # Full eval
    val_metrics = evaluate_split(model, val_loader, device, save_dir, "val", thr)
    test_metrics = evaluate_split(model, test_loader, device, save_dir, "test", thr)

    # Save metrics
    import json
    with open(save_dir / "metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    print("Done. Metrics saved to", save_dir)

if __name__ == "__main__":
    main()