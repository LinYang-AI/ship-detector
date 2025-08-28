import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

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
        manifest_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        use_cache: bool = False
    ):
        """
        Args:
            manifest_df: DataFrame with patch_path and has_ship columns,
            transform: Torchvision transforms to apply
            use_cache: Cache images in memory (faster but uses more RAM)
        """
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.use_cache = use_cache
        self.cache = {}
        
    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        
        # Load from cache or disk
        if self.use_cache and idx in self.cache:
            image = self.cache[idx]
        else:
            # Load image (handle both .tif and .jpg)
            img_path = row['patch_path']
            if img_path.endswith('.tif'):
                image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRA2RGB)
                elif image.shape[-1] == 4:
                    image = image[:, :, :3]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            
            if self.use_cache:
                self.cache[idx] = image
        
        # Convert to PIL for transforms
        image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(row['has_ship'], dtype=torch.float32)
        
        return image, label


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
        self.log('train-loss', loss, on_step=False, on_epoch=True, prog_bar=True)
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
    
    def on_train_epoch_end(self):
        # Optimizer
        opt_config =self.hparams['optimizer']
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
    
    # Training transforms with augmentation
    train_transforms = [
        transforms.RandomHorizontalFlip(p=aug_config.get('hflip_prob', 0.5)),
        transforms.RandomVerticalFlip(p=aug_config.get('vflip_prob', 0.5)),
    ]
    
    if aug_config.get('rotation', False):
        train_transforms.append(transforms.RandomRotation(degrees=90))
        
    if aug_config.get('color_jitter', False):
        train_transforms.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        )
    
    # Add normalization (ImageNet pretrained stats)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transforms.extend([
        transforms.ToTensor(),
        normalize
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)
    

def create_data_loader(
    manifest_path: str,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    df['has_ship'] = df['EncodedPixels'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    # Split data
    train_df, val_df = train_test_split(
        df,
        test_size=config['data']['val_split'],
        random_state=config['data']['random_seed'],
        stratify=df['has_ship']  # Maintain class distribution
    )
    
    print(f"Training samples: {len(train_df)}, (Ships: {train_df['has_ship'].sum()})")
    print(f"Validation samples: {len(val_df)}, (Ships: {val_df['has_ship'].sum()})")
    
    # Get transform
    train_transforms, val_transforms = get_augmentation_transforms(config)
    
    # Create datasets
    train_dataset = ShipPatchDataset(train_df, transform=train_transforms)
    val_dataset = ShipPatchDataset(val_df, transform=val_transforms)
    
    # Handle class imbalance with weighted sampling
    if config['training'].get('use_weighted_sampler', False):
        # Calculate weights for each sample
        train_labels = train_df['has_ship'].values
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    return train_loader, val_loader


def main(config_path: str, manifest_path: str, output_dir: str):
    """Main training function"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set seeds for reproducibility
    pl.seed_everything(config['data']['random_seed'])
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loader(manifest_path, config)
    
    # Initialize model
    model = ViTShipClassifier(config)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='vit-{epoch:02d}-{val_acc:.3f}',
            monitor='val_acc',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='vit_logs'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        precision=config['training'].get('precision', 32)
    )
    
    # Train
    print(f"\nStarting training with {config['model']['name']}")
    print(f"Output directory: {output_dir}")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = os.path.join(output_dir, 'vit_ship_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Print best checkpoint
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best validation accuracy: {callbacks[0].best_model_score:.4f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ViT for ship classification")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--manifest', type=str, required=True, help='Path to manifest CSV')
    parser.add_argument('--output-dir', type=str, default='./models/vit', help='Output directory')
    
    args = parser.parse_args()
    main(args.config, args.manifest, args.output_dir)
    