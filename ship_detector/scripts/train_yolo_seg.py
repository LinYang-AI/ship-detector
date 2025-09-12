import os
import yaml
from pathlib import Path
import logging
from typing import Literal, Optional

import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
from tqdm import tqdm
from ship_detector.scripts.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOShipSegmentation:
    """YOLOv8 segmentation model for ship detection"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to training configuration YAML
        """
        self.config = load_config(config_path)
        # Initialize model
        model_name = self.config['model']['architecture']
        if self.config['model']['pretrained']:
            # Load pretrained model
            self.model = YOLO(f"{model_name}.pt")
            logger.info(f"Loaded pretrained {model_name} model.")
        else:
            # Build from scratch
            self.model = YOLO(f"{model_name}.yaml")
            logger.info(f"Built {model_name} model from scratch.")
    
    def train(self, data_yaml: str, resume: bool = False):
        """Train the model
        
        Args:
            data_yaml: Path to dataset configuration file
            resume: Whether to resume from checkpoint
        """
        # Prepare training arguments
        train_args = {
            'data': data_yaml,
            'epochs': self.config['training']['epochs'],
            'patience': self.config['training']['patience'],
            'batch': self.config['data']['batch_size'],
            'imgsz': self.config['data']['image_size'],
            'save': True,
            'save_period': self.config['training']['save_period'],
            'cache': self.config['cache'],
            'device': [self.config['device']],
            'workers': self.config['data']['num_workers'],
            'project': self.config['paths']['project'],
            'name': self.config['paths']['name'],
            'exist_ok': True,
            'pretrained': self.config['model']['pretrained'],
            'optimizer': self.config['training']['optimizer'],
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': True,  # Single class (ship)
            'rect': self.config['rect'],
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': resume,
            'amp': self.config['amp'],
            'fraction': 1.0,
            'profile': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,

            # Hyperparamters
            'lr0': self.config['training']['lr0'],
            'lrf': self.config['training']['lrf'],
            'momentum': self.config['training']['momentum'],
            'weight_decay': self.config['training']['weight_decay'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'warmup_momentum': self.config['training']['warmup_momentum'],
            'warmup_bias_lr': self.config['training']['warmup_bias_lr'],
            'box': self.config['training']['box_loss_gain'],
            # 'seg': self.config['training']['seg_loss_gain'],
            'cls': self.config['training']['cls_loss_gain'],
            'dfl': self.config['training']['dfl_loss_gain'],
            
            # Augmentation
            'hsv_h': self.config['data']['augmentation']['hsv_h'],
            'hsv_s': self.config['data']['augmentation']['hsv_s'],
            'hsv_v': self.config['data']['augmentation']['hsv_v'],
            'degrees': self.config['data']['augmentation']['degrees'],
            'translate': self.config['data']['augmentation']['translate'],
            'scale': self.config['data']['augmentation']['scale'],
            'shear': self.config['data']['augmentation']['shear'],
            'perspective': self.config['data']['augmentation']['perspective'],
            'flipud': self.config['data']['augmentation']['flipud'],
            'fliplr': self.config['data']['augmentation']['fliplr'],
            'mosaic': self.config['data']['augmentation']['mosaic'],
            'mixup': self.config['data']['augmentation']['mixup'],
            'copy_paste': self.config['data']['augmentation']['copy_paste']
        }
        
        # Train model
        logger.info("Starting training...")
        results = self.model.train(**train_args)
        
        logger.info(f"Training completed. Best model saved to {results.saved_dir}")
        
    def validate(self, data_yaml: str, weights_path: Optional[str] = None):
        """Validate the model.
        
        Args:
            data_yaml: Path to dataset configuration
            weights_path: Path to model weights (if not using current model)
        """
        if weights_path:
            self.model = YOLO(weights_path)
            logger.info(f"Loaded weights from {weights_path}")
        
        # Validate
        metrics = self.model.val(
            data=data_yaml,
            imgsiz=self.config['data']['image_size'],
            batch=self.config['data']['batch_size'],
            conf=self.config['inference']['conf_thresh'],
            iou=self.config['inference']['iou_thresh'],
            device=self.config['device'],
            single_cls=True,
            save_json=True,
            save_hybrid=True,
            max_det=self.config['inference']['max_det'],
            half=self.config['amp'],
            dnn=False,
            plots=True,
            rect=self.config['rect'],
            split='val'
        )
        
        logger.info("Validation Results:")
        logger.info(f"\tBox mAP50: {metrics.box.map50:.4f}")
        logger.info(f"\tBox mAP50-95: {metrics.box.map:4f}")
        logger.info(f"\tMask mAP50: {metrics.seg.map50:.4f}")
        logger.info(f"\tMask mAP50-95: {metrics.seg.map:.4f}")
        
        return metrics
    
    def predict(self, image_path: str, conf_thresh: float = None, iou_thresh: float = None):
        """Run inference on a single image.
        
        Args:
            image_path: Path to input image
            conf_thresh: Confidence threshold
            iou_thresh: IoU threshold for NMS
            
        Returns:
            Prediction results
        """
        conf = conf_thresh or self.config['inference']['conf_thresh']
        iou = iou_thresh or self.config['inference']['iou_thresh']
        
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            imgsz=self.config['data']['image_size'],
            max_det=self.config['inference']['max_det'],
            device=self.config['device'],
            retina_masks=self.config['inference']['retina_masks'],
            classes=[0],
            agnostic_nms=False,
            save=False,
            save_txt=False,
            save_config=False,
            save_crop=False,
            show=False,
            stream=False,
        )
        
        return results
    
    def export_model(self, format: str = 'onnx'):
        """Export model to different formats.
        
        Args:
            format: Export format (onnx, torchscript, etc.)
        """
        logger.info(f"Exporting model to {format} format...")
        
        exported_model = self.model.export(
            format=format,
            imgsz=self.config['data']['image_size'],
            keras=False,
            optimize=True,
            half=False,
            int8=False,
            dynamic=True,
            simplify=True,
            opset=12,
            workspace=4,
            nms=False,
            batch=1
        )
        
        logger.info(f"Model exported to {exported_model}")
        return exported_model
    
def visualize_predictions(image_path: str, results, output_path: str):
    """Visualize segmentation predictions.
    
    Args:
        image_path: Path to original image
        results: YOLO prediction results
        output_path: Path to save visualization
        """
    # Load image
    image = cv2.imread(image_path)
    overlay = image.copy()
    
    # Process results
    for r in results:
        if r.masks is not None:
            # Get masks
            masks = r.masks.data.copu().numpy()
            
            # Get boxes and scores
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            scores = r.boxes.conf.cpu().numpy if r.boxes is not None else []
            
            # Draw each mask
            for i, mask in enumerate(masks):
                # Resize mask to image size
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
                
                # Create colored mask
                color = (0, 255, 0)  # Green for ships
                colored_mask = np.zeros_lik(image)
                colored_mask[mask == 1] = color
                
                # Add to overlay
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                
                # Draw bounding box if available
                if i < len(boxes):
                    box = boxes[i].astype(int)
                    score = scores[i]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(image, f"Ship {score:.2f}", (box[0], box[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Combine image and overlay
    result = cv2.addWeighted(image, 0.5, overlay, 0.4, 0)
    
    # Save result
    cv2.imwrite(output_path, result)
    logger.info(f"Saved visualization to {output_path}")
    

