import os
import argparse
from pathlib import Path
import ruamel.yaml as yaml
from typing import Dict, Any, Optional, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import timm
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ShipPatchDataset(Dataset):
    """Dataset for ship detection patches"""

    def __init__(
        self,
        config: Dict[str, Any],
        manifest_df: pd.DataFrame,
        transform: transforms.Compose = None,
        is_training: bool = True,
        use_cache: bool = False
    ):
        """
        Args:
            manifest_df: DataFrame with patch_path and has_ship columns,
            transform: Torchvision transforms to apply
            use_cache: Cache images in memory (faster but uses more RAM)
        """
        self.manifest = manifest_df.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.is_training = is_training
        self.preprocessing_method = config['model'].get(
            'preprocessing_method', 'adaptive')
        self.use_cache = use_cache
        self.cache = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        # Load from cache or disk
        image = cv2.imread(row['patch_path'])
        if image is None:
            raise FileNotFoundError(f"Image not found: {row['patch_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Check if it's actually 768x768
        if image.shape[:2] != (768, 768):
            image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_AREA)
        # Apply ship-preserving preprocessing
        image = self.preprocess_image(image)
        # Convert to PIL for standard transforms
        image = Image.fromarray(image)
        # transforms.Compose or single callable
        if self.transform:
            if isinstance(self.transform, list):
                # If mistakenly passed a list, compose it
                image = transforms.Compose(self.transform)(image)
            else:
                image = self.transform(image)
        label = torch.tensor(row['has_ship'], dtype=torch.float32)
        return image, label

        # if self.use_cache and idx in self.cache:
        #     image = self.cache[idx]
        # else:
        #     # Load image (handle both .tif and .jpg)
        #     img_path = row['patch_path']
        #     if img_path.endswith('.tif'):
        #         image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        #         if len(image.shape) == 2:
        #             image = cv2.cvtColor(image, cv2.COLOR_GRA2RGB)
        #         elif image.shape[-1] == 4:
        #             image = image[:, :, :3]
        #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     else:
        #         image = Image.open(img_path).convert('RGB')
        #         image = np.array(image)

        #     if self.use_cache:
        #         self.cache[idx] = image

        # # Convert to PIL for transforms
        # image = Image.fromarray(image.astype(np.uint8))

        # # Apply transforms
        # if self.transform:
        #     image = self.transform(image)

        # # Get label
        # label = torch.tensor(row['has_ship'], dtype=torch.float32)

        # return image, label

    def preprocess_image(self, image):
        """Apply ship-preserving preprocessing"""
        if self.preprocessing_method == 'multiscale':
            return self.multiscale_resize_with_context(image)
        elif self.preprocessing_method == 'adaptive':
            return self.adaptive_resize_preserve_ships(image)
        else:
            return cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    def multiscale_resize_with_context(self, image):
        """Multi-scale preprocessing"""
        h, w = image.shape[:2]

        # Full image view (global context)
        global_view = cv2.resize(
            image, (112, 112), interpolation=cv2.INTER_AREA)

        # Center crop view (detail preservation)
        center = h // 2
        crop_size = min(384, h)  # Handle smaller images
        start = max(0, center - crop_size // 2)
        end = min(h, start + crop_size)
        start = max(0, end - crop_size)  # Ensure crop_size

        center_crop = image[start:end, start:end]
        detail_view = cv2.resize(
            center_crop, (112, 112), interpolation=cv2.INTER_AREA)

        # Combine views
        combined = np.zeros((224, 224, 3), dtype=image.dtype)
        combined[:112, :112] = global_view
        combined[:112, 112:] = detail_view
        combined[112:, :] = cv2.resize(
            image, (224, 112), interpolation=cv2.INTER_AREA)

        return combined

    def adaptive_resize_preserve_ships(self, image):
        """Adaptive resize with ship preservation"""
        # Use INTER_AREA for better small object preservation
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Light sharpening to enhance edges
        if self.config['data'].get('apply_sharpening', False):
            strength = self.config['data'].get('sharpening_strength', 0.3)
            kernel = np.array([[-0.1, -0.1, -0.1],
                               [-0.1, 1 + 0.8 * strength, -0.1],
                               [-0.1, -0.1, -0.1]])
            sharpened = cv2.filter2D(resized, -1, kernel)
            resized = cv2.addWeighted(
                resized, 1 - strength, sharpened, strength, 0)

        return np.clip(resized, 0, 255).astype(np.uint8)


class ViTShipClassifier(pl.LightningModule):
    """Vision Transformer for ship classification"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)

        # Load pretrained ViT from timm
        self.model = timm.create_model(
            config['model']['name'],
            pretrained=config['model']['pretrained'],
            num_classes=1,  # Binary classification
        )

        # Optionally freeze backbone
        if config['model']['freeze_backbone_epochs'] > 0:
            self.freeze_backbone()

        # Loss function with optional class weights
        pos_weight = torch.tensor([config['training'].get('pos_weight', 1.0)])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Metric tracking
        self.train_acc = []
        self.val_acc = []

    def freeze_backbone(self):
        """Freeze all layers except the classification head"""
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.required_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = torch.sigmoid(outputs) > 0.5
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log('train-loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        probs = torch.sigmoid(outputs)
        preds = probs > 0.5
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'preds': preds, 'labels': labels, 'probs': probs}

    def configure_optimizers(self):
        # Optimizer
        opt_config = self.hparams['optimizer']
        if opt_config['name'] == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['name'] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")

        # Learning rate scheduler
        scheduler_config = self.hparams['scheduler']
        if scheduler_config['name'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config['T_max'],
                eta_min=scheduler_config.get('eta_min', 0)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif scheduler_config['name'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-7)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer


def get_augmentation_transforms(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create augmentation pipelines for training and validation."""
    aug_config = config['augmentation']

    # Training transforms
    train_transforms = []

    # Geometric augmentations (ship-aware)
    train_transforms.extend([
        transforms.RandomHorizontalFlip(p=aug_config.get('hflip_prob', 0.5)),
        transforms.RandomVerticalFlip(p=aug_config.get('vflip_prob', 0.5)),
    ])

    if aug_config.get('rotation', False):
        train_transforms.append(transforms.RandomRotation(degrees=90))

    # Color augmentations (conservative for ship detection)
    if aug_config.get('color_jitter', False):
        brightness = aug_config.get('brightness_range', [0.8, 1.2])
        contrast = aug_config.get('contrast_range', [0.9, 1.1])
        train_transforms.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=0.1,
                hue=0.05
            )
        )

    # Additional augmentations for robustness
    if aug_config.get('gaussian_blur_prob', 0) > 0:
        train_transforms.append(
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=aug_config['gaussian_blur_prob'])
        )

    # Standard preprocessing
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation transforms (no augmentation)
    val_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def create_data_loader(
    manifest_path: str,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""

    # Load manifest
    df = pd.read_csv(manifest_path)
    df['has_ship'] = df['EncodedPixels'].apply(
        lambda x: 0 if pd.isna(x) else 1)
    df['patch_path'] = df['ImageId'].apply(
        lambda x: f"data/airbus-ship-detection/train_v2/{x}")

    # Filter very small ships if needed
    if config['data'].get('min_ship_pixel_ratio'):
        # This would require additional preprocessing to calculate ship size
        pass

    # Split data
    train_df, val_df = train_test_split(
        df,
        test_size=config['data']['val_split'],
        random_state=config['data']['random_seed'],
        stratify=df['has_ship']  # Maintain class distribution
    )

    print(f"Training samples: {len(train_df)}")
    print(
        f"  - With ships: {train_df['has_ship'].sum()} ({train_df['has_ship'].mean()*100:.1f}%)")
    print(f"Validation samples: {len(val_df)}")
    print(
        f"  - With ships: {val_df['has_ship'].sum()} ({val_df['has_ship'].mean()*100:.1f}%)")

    # Get transform
    train_transforms, val_transforms = get_augmentation_transforms(config)

    # Create datasets
    train_dataset = ShipPatchDataset(
        train_df, config, transform=train_transforms, is_training=True)
    val_dataset = ShipPatchDataset(
        val_df, config, transform=val_transforms, is_training=False)

    # Handle class imbalance with weighted sampling
    if config['training'].get('use_weighted_sampler', False):
        # Calculate weights for each sample
        train_labels = train_df['has_ship'].values
        class_counts = np.bincount(train_labels)

        base_weight = 1.0 / class_counts
        # Give extra weight to positive samples (ships are harder to detect when small)
        ship_boost = config['training'].get('pos_weight', 1.0)
        class_weight = base_weight.copy()
        class_weight[1] *= ship_boost

        sample_weights = class_weight[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False  # Sampler handles randomization
    else:
        sampler = None
        shuffle = True

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader
