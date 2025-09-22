# Pneumonia Detection from Chest X-rays (ResNet-18)

Single-file training & inference with built-in Grad-CAM. Works on CPU.

## Quickstart (Windows / PowerShell)

```powershell
# 0) clone / open a terminal in the repo folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) Install PyTorch (CPU only)
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# 2) Install the rest
pip install -r requirements.txt
