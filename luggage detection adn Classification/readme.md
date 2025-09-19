# Luggage Detection & Material Classification

Detect luggage and classify by **material** and **shape** using **YOLOv8** (detection/materials) and **ResNet-50** (shape).

## Overview
- **Detection:** YOLOv8 (conf-thresholded boxes, resized to 416×416).
- **Shape:** ResNet-50 on 224×224 crops (regular vs. irregular).
- **Materials:** Hard-Plastic, Soft-Plastic, Cardboard, Others.
- **Optional:** Size estimation from bbox pixels → cm (calibrated factor).

## Data
- 623+ top-down images.
- Shape split 75/15/10; Material set expanded to 578 images with augmentation.
- Labeling & augmentation via **Roboflow**.

## Tech Stack
PyTorch · Ultralytics YOLOv8 · OpenCV · Google Colab · Roboflow


## Images
![Pipeline](assets/pipeline.png)
![Detections](assets/yolo_detections.jpg)
![Loss Curves](assets/training_loss.png)
