import os
from pathlib import Path
from collections import Counter

ROOT = Path("data/chest_xray")
splits = ["train","val","test"]
classes = ["NORMAL","PNEUMONIA"]

for sp in splits:
    for cls in classes:
        p = ROOT / sp / cls
        if not p.exists():
            print(f"Missing: {p}")
        else:
            n = len(list(p.glob("*.jpeg"))) + len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))
            print(f"{sp}/{cls}: {n} images")