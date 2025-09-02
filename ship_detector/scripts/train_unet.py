import os
import argparse
import ruamel.yaml as yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import segmentation_models_pytorch as smp
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class ShipSegmentationDataset(Dataset):
    """Dataset for ship segmentation (only positive patches)."""
    
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        only_ship_patches: bool = True,
        mask_dir: Optional[str] = None
    ):
        """
        Args:
            manifest_df: DataFrame with patch_path and has_ship columns
            transform: Albumentations transforms
            only_ship_patches: If True, only use patches with ships
            mask_dir: Directory containing mask files (if not in manifest)
        """
        if only_ship_patches:
            self.manifest = manifest_df[manifest_df['has_ship'] == 1].reset_index(drop=True)
            print(f"Using {len(self.manifest)} ship patches for segmentation")
        else:
            self.manifest = manifest_df.reset_index(drop=True)
        
        self.transform = transform
        self.mask_dir = mask_dir
        
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        
        # Load image
        img_path = row['patch_path']
        if img_path.endswith('.tif'):
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[-1] == 4:
                image = image[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        if 'mask_path' in row and pd.notna(row['mask_path']):
            mask_path = row['mask_path']
        else:
            # Construct mask path
            mask_name = Path(img_path).stem + '_mask.png'
            mask_path = os.path.join(self.mask_dir or Path(img_path).parent, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)  # Binary mask
        else:
            # No mask = no ships (for negative samples)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask has channel dimension
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for segmentation."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE Loss
        bce_loss = self.bce(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


class UNetShipSegmentation(pl.LightningModule):
    """U-Net model for ship segmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Create U-Net with pretrained encoder
        self.model = smp.Unet(
            encoder_name=config['model']['encoder'],
            encoder_weights=config['model']['encoder_weights'],
            in_channels=3,
            classes=1,
            activation=None  # Raw logits for loss
        )
        
        # Loss function
        self.criterion = DiceBCELoss(
            dice_weight=config['loss']['dice_weight'],
            bce_weight=config['loss']['bce_weight']
        )
        
        # Freezing configuration
        self.freeze_encoder = config['model'].get('freeze_encoder', False)
        self.unfreeze_epoch = config['model'].get('unfreeze_epoch', 5)
        self.freeze_bn = config['model'].get('freeze_bn', True)
        self.freeze_strategy = config['model'].get('freeze_strategy', 'full')  # 'full', 'partial'
        
        # Apply initial freezing
        if self.freeze_encoder:
            self._apply_freeze_strategy()
        
        # Log model statistics
        self._log_model_stats()
        
        # Metrics
        self.train_iou = []
        self.val_iou = []

    def _apply_freeze_strategy(self):
        """Apply the selected freezing strategy"""
        if self.freeze_strategy == 'full':
            self._freeze_encoder_full()
        elif self.freeze_strategy == 'partial':
            self._freeze_encoder_partial()
        elif self.freeze_strategy == 'gradual':
            self._freeze_encoder_gradual()
        else:
            logger.warning(f"Unknown freeze strategy: {self.freeze_strategy}")
    
    def _freeze_encoder_full(self):
        """Freeze entire encoder."""
        logger.info("Freezing the entire encoder.")
        # Freeze encoder parameters
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Optionally freeze batch norm layers
        if self.freeze_bn:
            self._freeze_batchnorm()
    
    def _freeze_encoder_partial(self):
        """Freeze only early layers of encoder (keep later layers trainable)"""
        logger.info("Freezing partial encoder (first 3/4 of layers)")
        
        # Get encoder stages
        encoder_children = list(self.model.encoder.children())
        
        # Freeze first 75% of encoder stages
        freeze_until = int(len(encoder_children) * 0.75)
        
        for i, child in enumerate(encoder_children):
            if i < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False
                logger.info(f"Froze encoder stage {i}")
            else:
                logger.info(f"Kept encoder stage {i} trainable")
    
    def _freeze_encoder_gradual(self):
        """Freeze encoder with gradual unfreezing schedule."""
        logger.info("Using gradual unfreezing strategy")
        
        # Start with everything frozen
        self._freeze_encoder_full()
        
        # Will unfreeze gradually during training
        self.layers_to_unfreeze = list(reversed(list(self.model.encoder.children())))
        
    
    def _freeze_batchnorm(self):
        """Freeze batch normalization layers in encoder."""
        logger.info("Freezing BatchNorm layers in encoder")
        
        def set_bn_eval(module):
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                # Disable gradient computation
                for param in module.parameters():
                    param.requires_grad = False
        
        self.model.encoder.apply(set_bn_eval)

    def _unfreeze_gradual_next_layer(self):
        """Unfreeze next layer in gradual strategy."""
        if hasattr(self, 'layers_to_unfreeze') and self.layers_to_unfreeze:
            layer = self.layers_to_unfreeze.pop(0)
            for param in layer.parameters():
                param.requires_grad = True
            logger.info(f"Unfroze layer at epoch {self.current_epoch}")
    
    def _log_model_stats(self):
        """Log model parameter statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
        
        logger.info("="*50)
        logger.info("Model Parameter Statistics:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        logger.info(f"Encoder parameters: {encoder_params:,} ({encoder_trainable:,} trainable)")
        logger.info(f"Decoder parameters: {decoder_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.1f}%")
        logger.info("="*50)

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Handle unfreezing strategies
        if self.freeze_encoder:
            if self.freeze_strategy == 'full' and self.current_epoch == self.unfreeze_epoch:
                logger.info(f"Reached unfreeze epoch {self.unfreeze_epoch}")
                self._unfreeze_eoncoder()
                self._log_model_stats()

            elif self.freeze_strategy == 'gradual':
                # Unfreeze one layer every few epochs
                if self.current_epoch > 0 and self.current_epoch % 2 == 0:
                    self._unfreeze_gradual_next_layer()
                    self._log_model_stats()
        
        # Ensure batch norm layers stay in eval model if frozen
        if self.freeze_bn:
            def set_bn_eval(module):
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
            self.model.encoder.apply(set_bn_eval)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Intersection over Union."""
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        
        intersection = (pred_binary * target).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()
    
    def calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice coefficient."""
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        
        intersection = (pred_binary * target).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return dice.mean()
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        iou = self.calculate_iou(outputs, masks)
        dice = self.calculate_dice(outputs, masks)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=False, on_epoch=True)
        
        # Log learning rates for different parameter groups
        if self.trainer.optimizers:
            opt = self.trainer.optimizers[0]
            for i, pg in enumerate(opt.param_groups):
                self.log(f"lr_group_{i}", pg['lr'], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        iou = self.calculate_iou(outputs, masks)
        dice = self.calculate_dice(outputs, masks)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True)
        
        # Log sample predictions every N epochs
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self.log_predictions(images, masks, outputs)
        
        return {'loss': loss, 'iou': iou, 'dice': dice}
    
    def log_predictions(self, images, masks, outputs, num_samples=4):
        """Log sample predictions to TensorBoard."""
        # Take first N samples
        n = min(num_samples, images.shape[0])
        
        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images_vis = images[:n] * std + mean
        images_vis = torch.clamp(images_vis, 0, 1)
        
        # Convert predictions to binary
        preds = (torch.sigmoid(outputs[:n]) > 0.5).float()
        
        # Create grid for visualization
        # Stack: [image, ground_truth, prediction]
        vis_tensors = []
        for i in range(n):
            # Convert masks to 3-channel for visualization
            gt_vis = masks[i].repeat(3, 1, 1)
            pred_vis = preds[i].repeat(3, 1, 1)
            
            # Create overlay (red for ground truth, green for prediction)
            overlay = torch.zeros_like(images_vis[i])
            overlay[0] = masks[i, 0]  # Red channel for GT
            overlay[1] = preds[i, 0]  # Green channel for pred
            
            # Stack horizontally
            vis_row = torch.cat([images_vis[i], gt_vis, pred_vis, overlay], dim=2)
            vis_tensors.append(vis_row)
        
        # Stack all samples vertically
        vis_grid = torch.cat(vis_tensors, dim=1)
        
        # Log to TensorBoard
        self.logger.experiment.add_image(
            'predictions',
            vis_grid,
            self.current_epoch
        )
    
    def configure_optimizers(self):
        """Configure optimizers with different learning rates for encoder/decoder."""
        
        # Separate encoder and decoder parameters
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        # Also include segmentation head parameters with decoder
        head_params = []
        for name, module in self.model.named_children():
            if name not in ['encoder', 'decoder']:
                head_params.extend(list(module.parameters()))
        
        # Different learning rates for encoder and decoder
        lr = self.hparams['optimizer']['lr']
        encoder_lr_scale = self.hparams['optimizer'].get('encoder_lr_scale', 0.1)
        # Create parameter group
        param_groups = []
        
        if self.freeze_encoder and self.freeze_strategy == 'full':
            # If encoder is frozen, only add decoder params initially
            param_groups.append({
                'params': decoder_params + head_params,
                'lr': lr,
                'name': 'decoder'
            })
        else:
            param_groups.extend([
                {
                    'params': encoder_params,
                    'lr': lr * encoder_lr_scale,
                    'name': 'encoder'
                },
                {
                    'params': decoder_params + head_params,
                    'lr': lr,
                    'name': 'decoder'
                }
            ])
        
        opt_config = self.hparams['optimizer']
        if opt_config['name'] == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['name'] == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        
        # Scheduler
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
        elif scheduler_config['name'] == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt_config['lr'],
                epochs=self.hparams['training']['max_epochs'],
                steps_per_epoch=1000  # Will be updated
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }
        else:
            return optimizer


def get_augmentation_transforms(config: Dict[str, Any]) -> Tuple[A.Compose, A.Compose]:
    """Create augmentation pipelines for segmentation."""
    
    aug_config = config['augmentation']
    
    # Training transforms
    train_transforms = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
    ]
    
    # Add strong augmentations for small objects
    if aug_config.get('strong_aug', False):
        train_transforms.extend([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=1),
                A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ], p=0.3),
        ])
    
    # Color augmentations
    if aug_config.get('color_aug', True):
        train_transforms.extend([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.RandomGamma(p=1),
            ], p=0.5),
        ])
    
    # Add noise for robustness
    if aug_config.get('noise_aug', True):
        train_transforms.extend([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(p=1),
                A.MultiplicativeNoise(p=1),
            ], p=0.2),
        ])
    
    # Normalization
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transforms.extend([
        normalize,
        ToTensorV2()
    ])
    
    # Validation transforms (only normalization)
    val_transforms = [
        normalize,
        ToTensorV2()
    ]
    
    return A.Compose(train_transforms), A.Compose(val_transforms)


def handle_multi_ship_instances(mask: np.ndarray, min_ship_size: int = 10) -> np.ndarray:
    """Process mask to handle multiple ship instances.
    
    Args:
        mask: Binary mask
        min_ship_size: Minimum pixels for valid ship
    
    Returns:
        Processed mask with small objects removed
    """
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    # Filter small objects
    filtered_mask = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id)
        if component_mask.sum() >= min_ship_size:
            filtered_mask[component_mask] = 1
    
    return filtered_mask


def create_data_loaders(
    manifest_path: str,
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders for segmentation."""
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    df['has_ship'] = df['EncodedPixels'].notna().astype(int)
    df['patch_path'] = df['ImageId'].apply(
        lambda x: f"data/airbus-ship-detection/train_v2/{x}"
    )
    # Filter for ship patches only
    ship_df = df[df['has_ship'] == 1].copy()
    
    if len(ship_df) == 0:
        raise ValueError("No ship patches found in manifest!")
    
    print(f"Found {len(ship_df)} patches with ships")
    
    # Split data
    train_df, val_df = train_test_split(
        ship_df,
        test_size=config['data']['val_split'],
        random_state=config['data']['random_seed']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Get transforms
    train_transform, val_transform = get_augmentation_transforms(config)
    
    # Create datasets
    train_dataset = ShipSegmentationDataset(
        train_df,
        transform=train_transform,
        only_ship_patches=True
    )
    
    val_dataset = ShipSegmentationDataset(
        val_df,
        transform=val_transform,
        only_ship_patches=True
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True  # For batch norm stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def main(config_path: str, manifest_path: str, output_dir: str):
    """Main training function."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seeds
    pl.seed_everything(config['data']['random_seed'])
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(manifest_path, config)
    
    # Initialize model
    model = UNetShipSegmentation(config)
    
    # Print model info
    print(f"\nModel: U-Net with {config['model']['encoder']} encoder")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='unet-{epoch:02d}-{val_iou:.3f}',
            monitor='val_iou',
            mode='max',
            save_top_k=3,
            save_last=True
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
        name='unet_logs'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        precision=config['training'].get('precision', 32)
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"Output directory: {output_dir}")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = os.path.join(output_dir, 'unet_ship_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")
    
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best validation IoU: {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for ship segmentation")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--manifest', type=str, required=True, help='Path to data manifest CSV')
    parser.add_argument('--output-dir', type=str, default='./models/unet', help='Output directory')
    
    args = parser.parse_args()
    main(args.config, args.manifest, args.output_dir)