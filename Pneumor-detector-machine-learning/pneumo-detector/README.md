# Pneumonia Detection from Chest X-rays

End‑to‑end PyTorch project to classify **Pneumonia vs. Normal** on the Kaggle Chest X-Ray Pneumonia dataset.

## 0) Quickstart

```bash
# 0. Create and activate env (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 1. Install deps
pip install -r requirements.txt

# 2. Download dataset via Kaggle CLI (you need kaggle.json)
#    How to set up: https://www.kaggle.com/docs/api
mkdir -p data && cd data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d .
cd ..

# 3. Train baseline (DenseNet-121)
python -m src.train --config configs/baseline_densenet121.yaml

# 4. Evaluate on test (automatically runs at end of training)
#    Metrics & artifacts saved under outputs/baseline_densenet121

# 5. Run single-image inference + Grad-CAM overlay
python -m src.infer --ckpt outputs/baseline_densenet121/best.ckpt   --image data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg   --overlay out_overlay.png
```

## Project layout
```
pneumo-detector/
  ├─ configs/
  │   ├─ baseline_resnet18.yaml
  │   └─ baseline_densenet121.yaml
  ├─ src/
  │   ├─ datasets.py
  │   ├─ models.py
  │   ├─ train.py
  │   ├─ evaluate.py
  │   ├─ explain.py
  │   └─ infer.py
  ├─ scripts/
  │   └─ verify_kaggle_split.py
  ├─ requirements.txt
  └─ README.md
```

## Notes
- This code expects the Kaggle folder structure: `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`.
- Threshold is chosen on **val** set to achieve target sensitivity (default 0.90), then fixed for test.
- Outputs include: ROC/PR curves, confusion matrix, classification report, Grad‑CAM panels, and best checkpoint.
- Educational use only — **not for clinical use**.