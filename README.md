# Ship Detection and Segmentation Pipeline

## 📋 Project Overview

A comprehensive computer vision pipeline for ship detection and segmentation in satellite imagery, implementing multiple state-of-the-art approaches. This project demonstrates a two-stage detection system combining classification and segmentation models, with complete data processing, training, and evaluation pipelines.
**Dataset**: Airbus Ship Detection Challenge
**Task**: Detect and segment ships in satellite images using RLE-encoded masks
**Approach**: Two-stage pipeline with modular architecture supporting multiple model backends

## 🎯 Key Features

- **Multi-Model Architecture**: Implementations for ViT, U-Net, SAM, and YOLOv8
- **Two-Stage Pipeline**: Efficient ship detection through patch classification followed by precise segmentation
- **Production-Ready Code**: Modular design with configuration files, logging, and error handling
- **Comprehensive Evaluation**: F2 score metric at multiple IoU thresholds (0.5-0.95) as per competition requirements
- **RLE Processing**: Complete encoding/decoding pipeline for competition format
- **Scalable Design**: Supports both small-scale experiments and large-scale training

🏗️ Architecture

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

# 🚀 Implementation Status

## ✅ Completed Components

### Data Pipeline

 RLE mask encoding/decoding with column-major order handling
 Intelligent patch generation with overlap and boundary handling
 Geospatial metadata preservation
 Data augmentation strategies for maritime imagery
 Synthetic data generation for testing

### Stage 1: ViT Classifier

 Vision Transformer implementation with PyTorch Lightning
 Transfer learning from ImageNet pretrained models
 Mixed precision training support
 Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1)
 Trained for initial epochs - achieving 97%+ validation accuracy
