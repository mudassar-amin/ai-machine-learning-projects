from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from pathlib import Path

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(1).detach().cpu().numpy()
        logits_list.append(logits)
        labels_list.append(y.numpy())
    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)
    probs = _sigmoid(logits)
    return logits, probs, labels

def pick_threshold_for_sensitivity(labels, probs, target_sens=0.90):
    # sweep thresholds from 0..1 and return threshold hitting >= target_sens with max specificity
    fpr, tpr, thr = roc_curve(labels, probs)
    # tpr is sensitivity
    mask = tpr >= target_sens
    if not mask.any():
        # fallback: highest tpr
        idx = tpr.argmax()
    else:
        # among those, pick minimal fpr (max specificity)
        idx = np.argmin(fpr[mask])
        # need global index
        idx = np.arange(len(thr))[mask][idx]
    chosen_thr = thr[idx]
    return float(chosen_thr)

def plot_and_save_curves(labels, probs, out_dir: Path, split_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {split_name}")
    plt.legend(loc="lower right")
    plt.savefig(out_dir / f"roc_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    plt.figure()
    plt.plot(rec, prec, label=f"AUPRC={auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {split_name}")
    plt.legend(loc="lower left")
    plt.savefig(out_dir / f"pr_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

def save_confusion_and_report(labels, probs, threshold, out_dir: Path, split_name: str):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0,1])
    # Save confusion matrix plot
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {split_name}")
    plt.xticks([0,1], ["Normal","Pneumonia"])
    plt.yticks([0,1], ["Normal","Pneumonia"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(out_dir / f"confusion_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Classification report
    report = classification_report(labels, preds, target_names=["Normal","Pneumonia"], digits=4)
    with open(out_dir / f"classification_report_{split_name}.txt", "w") as f:
        f.write(report)
    return cm, report

def evaluate_split(model, loader, device, out_dir: Path, split_name: str, threshold: float):
    logits, probs, labels = collect_logits(model, loader, device)
    plot_and_save_curves(labels, probs, out_dir, split_name)
    cm, report = save_confusion_and_report(labels, probs, threshold, out_dir, split_name)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return {"AUROC": float(auroc), "AUPRC": float(auprc), "threshold": float(threshold)}