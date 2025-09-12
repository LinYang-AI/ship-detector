# Ship Detection and Segmentation Pipeline

## 📋 Project Overview

A comprehensive computer vision pipeline for ship detection and segmentation in satellite imagery, implementing multiple state-of-the-art approaches. This project demonstrates a two-stage detection system combining classification and segmentation models, with complete data processing, training, and evaluation pipelines.

 **Dataset** : Airbus Ship Detection Challenge from Kaggle: https://www.kaggle.com/competitions/airbus-ship-detection

 **Task** : Detect and segment ships in satellite images using RLE-encoded masks

 **Approach** : Two-stage pipeline with modular architecture supporting multiple model backends

## 🎯 Key Features

* **Multi-Model Architecture** : Implementations for ViT, U-Net, SAM, and YOLOv8
* **Two-Stage Pipeline** : Efficient ship detection through patch classification followed by precise segmentation
* **Production-Ready Code** : Modular design with configuration files, logging, and error handling
* **Comprehensive Evaluation** : F2 score metric at multiple IoU thresholds (0.5-0.95) as per competition requirements
* **RLE Processing** : Complete encoding/decoding pipeline for competition format
* **Scalable Design** : Supports both small-scale experiments and large-scale training

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Satellite Image │ --> │ Patch Generation │ --> │ ViT Classifier  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                   ┌───────────────┐
                                                   │ Ship Patches  │
                                                   └───────────────┘
                                                           │
                            ┌──────────────────────────────┼──────────────────────────────┐
                            ▼                              ▼                              ▼
                    ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
                    │    U-Net     │              │     SAM      │              │    YOLOv8    │
                    │ Segmentation │              │ Zero/Few-shot│              │  Real-time   │
                    └──────────────┘              └──────────────┘              └──────────────┘
                            │                              │                              │
                            └──────────────────────────────┼──────────────────────────────┘
                                                           ▼
                                                   ┌───────────────┐
                                                   │  RLE Masks    │
                                                   └───────────────┘
```

## 🚀 Implementation Status

### ✅ Completed Components

#### Data Pipeline

* [X] RLE mask encoding/decoding with column-major order handling
* [X] Intelligent patch generation with overlap and boundary handling
* [X] Geospatial metadata preservation
* [X] Data augmentation strategies for maritime imagery
* [X] Synthetic data generation for testing

#### Stage 1: ViT Classifier

* [X] Vision Transformer implementation with PyTorch Lightning
* [X] Transfer learning from ImageNet pretrained models
* [X] Mixed precision training support
* [X] Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1)
* [X] **Trained for initial epochs** - achieving 97%+ validation accuracy

#### Stage 2: Segmentation Models

**U-Net Implementation**

* [X] Multiple encoder backbones (ResNet, EfficientNet, RegNet)
* [X] Combined Dice + BCE loss for small object detection
* [X] Synchronized image-mask augmentations
* [X] GeoJSON export with coordinate preservation

**SAM (Segment Anything Model) Integration**

* [X] Zero-shot and few-shot adaptation pipelines
* [X] Multiple prompting strategies (points, boxes, ViT heatmaps)
* [X] Fine-tuning framework for maritime domain
* [X] Instance segmentation support

**YOLOv8 Segmentation**

* [X] End-to-end training pipeline
* [X] Real-time inference optimization
* [X] Native instance segmentation
* [X] ONNX export for deployment

#### Evaluation Framework

* [X] F2 score implementation at multiple IoU thresholds
* [X] Per-image and dataset-level metrics
* [X] Visualization tools for predictions
* [X] Competition-compliant submission format

🔄 Training Status

| Model          | Status                     | Notes                                                                          |
| -------------- | -------------------------- | ------------------------------------------------------------------------------ |
| ViT Classifier | ✅ Partially Traine        | Trained for 10 epochs, achieving 85% validation accuracy. Ready for inference. |
| U-Net          | 📝 Implementation Complete | Full training pipeline implemented. Requires GPU resources for training.       |
| SAM            | 📝 Implementation Complete | Zero-shot inference ready. Fine-tuning pipeline implemented.                   |
| YOLOv8         | 📝 Implementation Complete | Training configuration ready. Awaiting computational resources                 |

## 📊 Preliminary Results

### ViT Classifier Performance

```
Epoch 17/40:
├── Training Accuracy: 97.6%
├── Validation Accuracy: 97.6%
├── Training Loss: 0.11
|── Validation Loss: 0.09
└── Training Speed: 28,600 patches/second (RTX 5060Ti 16Gb)
```

*Note: Full training was limited by computational resources. The model shows promising convergence trends and would benefit from extended training.*

### 🛠️ Technical Stack

* **Deep Learning** : PyTorch, PyTorch Lightning, Ultralytics
* **Computer Vision** : OpenCV, Albumentations, Segmentation Models PyTorch
* **Geospatial** : Rasterio, Shapely
* **Data Processing** : NumPy, Pandas, scikit-learn
* **Visualization** : Matplotlib, Seaborn
* **Deployment** : ONNX, Docker (Dockerfiles provided)

### 💻 Installation

```bash
# Clone repository
git clone https://github.com/LinYang-AI/ship-detector.git
cd ship-detection-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
```



### 🎮 Usage

upcoming...
