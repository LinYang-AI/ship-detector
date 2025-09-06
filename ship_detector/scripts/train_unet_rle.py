import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import warnings

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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# RLE Processing (from corrected pipeline)
# ============================================================================

class RLEProcessor:
    """Handle RLE encoding/decoding with proper synchronization."""
    
    @staticmethod
    def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
        """Decode RLE string to binary mask.
        
        Args:
            mask_rle: Run-length encoded string
            shape: (height, width) of target image
        
        Returns:
            Binary mask array
        """
        if pd.isna(mask_rle) or mask_rle == '' or mask_rle == '0':
            return np.zeros(shape, dtype=np.uint8)
        
        s = mask_rle.split()
        starts = np.array(s[0::2], dtype=int) - 1  # RLE uses 1-based indexing
        lengths = np.array(s[1::2], dtype=int)
        ends = starts + lengths
        
        # Create mask
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, end in zip(starts, ends):
            img[start:end] = 1
        
        # IMPORTANT: RLE uses column-major (Fortran) order
        return img.reshape(shape, order='F')
    
    @staticmethod
    def rle_encode(mask: np.ndarray) -> str:
        """Encode binary mask to RLE string."""
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        
        if len(runs) == 2 and runs[1] == len(pixels) - 1:
            return ''
        
        return ' '.join(str(x) for x in runs)


# ============================================================================
# Dataset with Proper Mask Handling
# ============================================================================

class ShipSegmentationDataset(Dataset):
    """Dataset for ship segmentation with correctly processed masks."""
    
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        only_ship_patches: bool = True,
        verify_alignment: bool = False,
        use_rle: bool = False,
        rle_csv_path: Optional[str] = None
    ):
        """
        Args:
            manifest_df: DataFrame with patch_path, mask_path, has_ship columns
            transform: Albumentations transforms
            only_ship_patches: If True, only use patches with ships
            verify_alignment: If True, verify mask alignment on first batch
            use_rle: If True, decode RLE on the fly (for original data)
            rle_csv_path: Path to RLE CSV if using on-the-fly decoding
        """
        if only_ship_patches:
            self.manifest = manifest_df[manifest_df['has_ship'] == 1].reset_index(drop=True)
            if len(self.manifest) == 0:
                logger.warning("No ship patches found! Using all patches.")
                self.manifest = manifest_df.reset_index(drop=True)
        else:
            self.manifest = manifest_df.reset_index(drop=True)
        
        logger.info(f"Dataset initialized with {len(self.manifest)} patches")
        
        self.transform = transform
        self.verify_alignment = verify_alignment
        self.use_rle = use_rle
        self.rle_processor = RLEProcessor()
        
        # Load RLE data if needed
        if use_rle and rle_csv_path:
            self.rle_df = pd.read_csv(rle_csv_path)
            self.rle_df = self.rle_df.set_index('ImageId')
        else:
            self.rle_df = None
        
        # Verify first sample
        if verify_alignment and len(self.manifest) > 0:
            self._verify_first_sample()
    
    def _verify_first_sample(self):
        """Verify alignment of first sample."""
        row = self.manifest.iloc[0]
        
        # Load image
        image = cv2.imread(row['patch_path'])
        if image is None:
            logger.error(f"Cannot load image: {row['patch_path']}")
            return
        
        # Load mask
        if 'mask_path' in row and pd.notna(row['mask_path']) and os.path.exists(row['mask_path']):
            mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
            
            # Verify shapes match
            if image.shape[:2] != mask.shape[:2]:
                logger.error(f"Shape mismatch: image {image.shape[:2]} != mask {mask.shape[:2]}")
            else:
                logger.info(f"✓ First sample verified: image and mask shapes match {image.shape[:2]}")
                
                # Save verification image
                self._save_verification_image(image, mask, "verification_sample.jpg")
    
    def _save_verification_image(self, image: np.ndarray, mask: np.ndarray, filename: str):
        """Save overlay image for verification."""
        overlay = image.copy()
        overlay[mask > 127] = [0, 0, 255]  # Red for mask
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Add contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        cv2.imwrite(filename, result)
        logger.info(f"Verification image saved to {filename}")
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        
        # Load image
        img_path = row['patch_path']
        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            # Return black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or generate mask
        mask = self._load_mask(row, image.shape[:2])
        
        # Verify alignment (critical!)
        if image.shape[:2] != mask.shape[:2]:
            logger.error(f"Shape mismatch at idx {idx}: image {image.shape[:2]} != mask {mask.shape[:2]}")
            # Resize mask to match image
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply synchronized transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default normalization if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()
        
        # Ensure mask has correct shape [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        elif mask.shape[0] != 1:
            mask = mask[:1]  # Take first channel only
        
        return image, mask
    
    def _load_mask(self, row: pd.Series, expected_shape: Tuple[int, int]) -> np.ndarray:
        """Load mask from file or RLE."""
        
        # Option 1: Load from preprocessed mask file
        if 'mask_path' in row and pd.notna(row['mask_path']) and os.path.exists(row['mask_path']):
            mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            return mask
        
        # Option 2: Decode from RLE if available
        if self.use_rle and self.rle_df is not None and 'source_image' in row:
            source_image = row['source_image']
            if source_image in self.rle_df.index:
                rle_data = self.rle_df.loc[source_image]
                
                # Handle multiple RLE strings for same image
                if isinstance(rle_data, pd.DataFrame):
                    combined_mask = np.zeros(expected_shape, dtype=np.float32)
                    for _, rle_row in rle_data.iterrows():
                        if pd.notna(rle_row['EncodedPixels']):
                            mask = self.rle_processor.rle_decode(
                                rle_row['EncodedPixels'], 
                                expected_shape
                            )
                            combined_mask = np.maximum(combined_mask, mask)
                    return combined_mask
                else:
                    if pd.notna(rle_data['EncodedPixels']):
                        return self.rle_processor.rle_decode(
                            rle_data['EncodedPixels'],
                            expected_shape
                        ).astype(np.float32)
        
        # Option 3: No mask (no ships in this patch)
        return np.zeros(expected_shape, dtype=np.float32)


# ============================================================================
# Loss Functions
# ============================================================================

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


# ============================================================================
# U-Net Model with Validation
# ============================================================================

class UNetShipSegmentation(pl.LightningModule):
    """U-Net model for ship segmentation with mask validation."""
    
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
        
        # Metrics tracking
        self.train_iou_history = []
        self.val_iou_history = []
        
        # Validation flag
        self.first_batch_validated = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate IoU and Dice metrics."""
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        
        # IoU
        intersection = (pred_binary * target).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        # Dice
        dice_intersection = 2 * (pred_binary * target).sum(dim=(1, 2, 3))
        dice_union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (dice_intersection + 1e-6) / (dice_union + 1e-6)
        
        return {
            'iou': iou.mean(),
            'dice': dice.mean()
        }
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        
        # Validate first batch shapes
        if not self.first_batch_validated:
            logger.info(f"First batch - Images: {images.shape}, Masks: {masks.shape}")
            assert images.shape[0] == masks.shape[0], "Batch size mismatch!"
            assert masks.shape[1] == 1, f"Mask should have 1 channel, got {masks.shape[1]}"
            self.first_batch_validated = True
        
        # Forward pass
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        metrics = self.calculate_metrics(outputs, masks)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', metrics['iou'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_dice', metrics['dice'], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        metrics = self.calculate_metrics(outputs, masks)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', metrics['iou'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', metrics['dice'], on_step=False, on_epoch=True)
        
        # Log sample predictions periodically
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_predictions(images, masks, outputs)
        
        return {'loss': loss, 'iou': metrics['iou'], 'dice': metrics['dice']}
    
    def _log_predictions(self, images, masks, outputs, num_samples=4):
        """Log sample predictions for visualization."""
        import matplotlib.pyplot as plt
        
        n = min(num_samples, images.shape[0])
        
        # Create figure
        fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        # Denormalize images for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images_vis = images[:n] * std + mean
        images_vis = torch.clamp(images_vis, 0, 1)
        
        # Convert predictions to binary
        preds = (torch.sigmoid(outputs[:n]) > 0.5).float()
        
        for i in range(n):
            # Original image
            img = images_vis[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            gt = masks[i, 0].cpu().numpy()
            axes[i, 1].imshow(gt, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            pred = preds[i, 0].cpu().numpy()
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Overlay
            overlay = img.copy()
            overlay[:, :, 0] = np.where(gt > 0.5, 1, overlay[:, :, 0])  # Red for GT
            overlay[:, :, 1] = np.where(pred > 0.5, 1, overlay[:, :, 1])  # Green for pred
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (R=GT, G=Pred)')
            axes[i, 3].axis('off')
        
        plt.suptitle(f'Epoch {self.current_epoch} Predictions')
        plt.tight_layout()
        
        # Save figure
        save_path = f"predictions_epoch_{self.current_epoch}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved predictions to {save_path}")
    
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
        
        return optimizer


# ============================================================================
# Data Loading with Validation
# ============================================================================

def get_augmentation_transforms(config: Dict[str, Any]) -> Tuple[A.Compose, A.Compose]:
    """Create augmentation pipelines for training and validation."""
    
    aug_config = config['augmentation']
    
    # Training transforms with augmentation
    train_transforms = []
    
    # Geometric transforms (apply to both image and mask)
    if aug_config.get('rotation', True):
        train_transforms.append(A.RandomRotate90(p=0.5))
    
    if aug_config.get('flip', True):
        train_transforms.append(A.HorizontalFlip(p=0.5))
        train_transforms.append(A.VerticalFlip(p=0.5))
    
    if aug_config.get('shift_scale_rotate', True):
        train_transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.5
            )
        )
    
    # Strong augmentations for small objects
    if aug_config.get('strong_aug', False):
        train_transforms.append(
            A.OneOf([
                A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05),
                A.GridDistortion(p=1),
                A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ], p=0.3)
        )
    
    # Color augmentations (only for image)
    if aug_config.get('color_aug', True):
        train_transforms.append(
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.RandomGamma(p=1),
            ], p=0.5)
        )
    
    # Normalization (always applied)
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


def create_data_loaders(
    manifest_path: str,
    config: Dict[str, Any],
    verify_first_batch: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders with verification."""
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(df)} entries")
    
    # Filter for ship patches if specified
    if config['data'].get('only_ship_patches', True):
        ship_df = df[df['has_ship'] == 1].copy()
        logger.info(f"Filtered to {len(ship_df)} ship patches")
        
        if len(ship_df) == 0:
            logger.warning("No ship patches found! Using all patches.")
            ship_df = df.copy()
    else:
        ship_df = df.copy()
    
    # Split data
    train_df, val_df = train_test_split(
        ship_df,
        test_size=config['data']['val_split'],
        random_state=config['data']['random_seed'],
        stratify=ship_df['has_ship'] if 'has_ship' in ship_df.columns else None
    )
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Get transforms
    train_transform, val_transform = get_augmentation_transforms(config)
    
    # Create datasets
    train_dataset = ShipSegmentationDataset(
        train_df,
        transform=train_transform,
        only_ship_patches=False,  # Already filtered
        verify_alignment=verify_first_batch
    )
    
    val_dataset = ShipSegmentationDataset(
        val_df,
        transform=val_transform,
        only_ship_patches=False,
        verify_alignment=False
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
    
    # Verify first batch
    if verify_first_batch:
        logger.info("Verifying first batch...")
        first_batch = next(iter(train_loader))
        images, masks = first_batch
        logger.info(f"✓ First batch loaded successfully")
        logger.info(f"  Images shape: {images.shape}")
        logger.info(f"  Masks shape: {masks.shape}")
        logger.info(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
        logger.info(f"  Mask range: [{masks.min():.2f}, {masks.max():.2f}]")
        logger.info(f"  Mask unique values: {torch.unique(masks).tolist()[:10]}...")
    
    return train_loader, val_loader


# ============================================================================
# Main Training Function
# ============================================================================

def main(config_path: str, manifest_path: str, output_dir: str, verify_data: bool = True):
    """Main training function with integrated RLE handling."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    pl.seed_everything(config['data']['random_seed'])
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders with verification
    train_loader, val_loader = create_data_loaders(
        manifest_path, 
        config,
        verify_first_batch=verify_data
    )
    
    # Initialize model
    model = UNetShipSegmentation(config)
    
    # Log model info
    logger.info(f"\nModel: U-Net with {config['model']['encoder']} encoder")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    logger_tb = TensorBoardLogger(
        save_dir=output_dir,
        name='unet_logs'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger_tb,
        log_every_n_steps=10,
        precision=config['training'].get('precision', 32)
    )
    
    # Train
    logger.info(f"\nStarting training...")
    logger.info(f"Output directory: {output_dir}")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = os.path.join(output_dir, 'unet_final.pt')
    torch.save(model.state_dict(), final_path)
    logger.info(f"\nTraining complete! Model saved to {final_path}")
    
    # Print best checkpoint
    logger.info(f"Best checkpoint: {callbacks[0].best_model_path}")
    logger.info(f"Best validation IoU: {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net with corrected RLE handling")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--manifest', type=str, required=True, help='Path to data manifest CSV')
    parser.add_argument('--output-dir', type=str, default='./models/unet', help='Output directory')
    parser.add_argument('--verify-data', action='store_true', help='Verify data alignment')
    
    args = parser.parse_args()
    main(args.config, args.manifest, args.output_dir, args.verify_data)