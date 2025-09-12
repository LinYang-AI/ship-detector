import os
import yaml
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# SAM imports
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide

from ship_detector.scripts.utils import load_config, rle_decode

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShipSAMDataset(Dataset):
    """Dataset for SAM fine-tuning on ship segmentation task."""
    
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        sam_transform: ResizeLongestSide,
        patch_size: int = 1024,
        use_vit_prompts: bool = False,
        vit_model_path: Optional[str] = None,
    ):
        """
        Args:
            manifest_df: DataFrame with image and mask paths.
            sam_transform: SAM's image transform
            patch_size: Size of patches (SAM prefers 1024)
            use_vit_prompts: Whether to use ViT classifier for prompts
            vit_model_path: Path to ViT classifier checkpoint
        """
        self.manifest = manifest_df[manifest_df['has_ship'] == 1].reset_index(drop=True)
        self.transform = sam_transform
        self.patch_size = patch_size
        self.use_vit_prompts = use_vit_prompts
        
        # Load ViT model if needed
        if use_vit_prompts and vit_model_path:
            self.vit_model = self._load_vit_model(vit_model_path)
        else:
            self.vit_model = None
            
        logger.info(f"Dataset initialized with {len(self.manifest)} ship patches")
        
    def _load_vit_model(self, model_path: str) -> nn.Module:
        """Load pretrained ViT classifier for prompt generation."""
        # Simplified - would load actual ViT model
        logger.info(f"Loading ViT model from {model_path}")
        return None  # Placeholder for actual model loading
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        print(row['patch_path'])
        image = cv2.imread(row['patch_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        image = cv2.imread(row['patch_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        gt_mask = rle_decode(row['EncodedPixels'], (height, width))
        # gt_mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRASCALE)
        # gt_mask = (gt_mask > 127).astype(np.unit8)
        
        # Resize to SAM input size
        if image.shape[0] != self.patch_size:
            image = cv2.resize(image, (self.patch_size, self.patch_size))
            gt_mask = cv2.resize(gt_mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            
        # Apply SAM transform
        transformed = self.transform.apply_image(image)
        
        # Generate prompts (points or boxes)
        prompts = self._generate_prompts(gt_mask, image)
        return {
            'image': torch.from_numpy(transformed).permute(2, 0, 1).float() / 255.0,
            'original_image': image,
            'gt_mask': torch.from_numpy(gt_mask).float(),
            'prompts': prompts,
            'image_path': row['patch_path']
        }
    
    def _generate_prompts(self, mask: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
        """Generate point and box prompts from ground truth mask."""
        prompts = {}
        
        # Find connected components (individual ships)
        num_labels, labels = cv2.connectedComponents(mask)
        
        point_coords = []
        point_labels = []
        boxes = []
        
        for label_id in range(1, num_labels):
            ship_mask = (labels == label_id).astype(np.uint8)
            
            # Get bounding box
            contours, _ = cv2.findContours(ship_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                boxes.append([x, y, x+w, y+h])
                
                # Get center point as positive prompt
                center_x = x + w // 2
                center_y = y + h // 2
                point_coords.append([center_x, center_y])
                point_labels.append(1)
                
                # Add negative points outside ship
                for _ in range(2):
                    neg_x = np.random.randint(0, self.patch_size)
                    neg_y = np.random.randint(0, self.patch_size)
                    if mask[neg_y, neg_x] == 0:
                        point_coords.append([neg_x, neg_y])
                        point_labels.append(0)
        
        prompts['point_coords'] = np.array(point_coords) if point_coords else None
        prompts['point_labels'] = np.array(point_labels) if point_labels else None
        prompts['points'] = (prompts['point_coords'], prompts['point_labels'])
        prompts['boxes'] = np.array(boxes) if boxes else None
        
        # Add ViT heatmap if available
        if self.vit_model is not None:
            prompts['vit_heatmap'] = self._get_vit_heatmap(image)
        
        return prompts

    def _get_vit_heatmap(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Generate attention heatmap from ViT classifier."""
        # Placeholder for actual ViT heatmap generation
        return None


class SAMShipSegmentation(pl.LightningModule):
    """Lightning module for SAM fine-tuning on ship detection."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Load SAM model
        self.sam = self._load_sam_model(config)
        
        # Freeze components based on config
        if config['finetune']['freeze_image_encoder']:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        if config['finetune']['freeze_prompt_encoder']:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False
        
        # Load functions
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()
        
        # Metrics tracking
        self.val_ious = []
    
    def _load_sam_model(self, config: Dict[str, Any]) -> nn.Module:
        """Load SAM model with specified checkpoint."""
        model_type = config['model']['checkpoint_type']
        checkpoint_path = config['model']['checkpoint_path']
        
        # Download checkpoint if needed
        if not os.path.exists(checkpoint_path):
            logger.info(f"Downloading SAM checkpoint to {checkpoint_path}")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(
                config['model']['checkpoint_url'],
                checkpoint_path
            )
            
        # Load model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        # state_dict = torch.load(checkpoint_path, map_location='cpu')
        # sam.load_state_dict(state_dict)
        return sam
    
    def forward(self, image: torch.Tensor, prompts: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass through SAM with prompts."""
        # Encode image
        image_embeddings = self.sam.image_encoder(image)
        logger.info(f"Image embeddings shape: {image_embeddings.shape}")
        logger.info(f"Prompts points 0 shape: {prompts.get('points')[0].shape if prompts.get('points') is not None else None}")
        logger.info(f"Prompts points 1 shape: {prompts.get('points')[1].shape if prompts.get('points') is not None else None}")
        print(f"\nImage embeddings shape: {image_embeddings.shape}")
        print(f"\nPrompts points 0 shape: {prompts.get('points')[0].shape if prompts.get('points') is not None else None}")
        print(f"\nPrompts points 1 shape: {prompts.get('points')[1].shape if prompts.get('points') is not None else None}")
        # raise("Debugging - remove this line later")
        # Encode prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=prompts.get('points'),
            boxes=prompts.get('boxes'),
            masks=prompts.get('mask_input')
        )
        
        # Predict masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.hparams['inference']['multimask_output']
        )
        
        # Upscale masks to original size
        masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode='bilinear',
            align_corners=False,
        )
        
        return {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_masks': low_res_masks
        }
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self(batch['image'], batch['prompts'])
        
        # Calculate losses
        gt_mask = batch['gt_mask'].unsqueeze(1)
        
        # Use best mask (highest IoU prediction)
        if outputs['masks'].shape[1] > 1: # Multiple masks
            best_idx = outputs['iou_predictions'].argmax(dim=1)
            masks = outputs['masks'][torch.arange(len(best_idx)), best_idx].unsqueeze(1)
        else:
            masks = outputs['masks']
        
        # Compute losses
        focal_loss = self.focal_loss(masks, gt_mask)
        dice_loss = self.dice_loss(masks, gt_mask)
        iou_loss = self.iou_loss(masks, gt_mask)
        
        # Combined loss
        total_loss = (
            self.hparams['finetune']['focal_loss_weight'] * focal_loss +
            self.hparams['finetune']['dice_loss_weight'] * dice_loss +
            self.hparams['finetune']['iou_loss_weight'] * iou_loss
        )
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True)
        self.log('train_focal', focal_loss, on_step=False, on_epoch=True)
        self.log('train_dice', dice_loss, on_step=False, on_epoch=True)
        self.log('train_iou', iou_loss, on_step=False, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        outputs = self(batch['image'], batch['prompts'])
        
        # Get best mask
        if outputs['masks'].shape[1] > 1:
            best_idx = outputs['iou_predictions'].argmax(dim=1)
            masks = outputs['masks'][torch.arange(len(best_idx)), best_idx].unsqueeze(1)
        else:
            masks = outputs['masks']
        
        # Calculate metrics
        gt_mask = batch['gt_mask'].unsqueeze(1)
        iou = self.calculate_iou(masks, gt_mask)
        
        self.val_ious.append(iou)
        
        # Log sample predictions
        if batch_idx == 0:
            self._log_predictions(batch, outputs)
            
        return {'iou': iou}

    def on_validation_epoch_end(self):
        avg_iou = torch.stack(self.val_ious).mean()
        self.log('val_iou', avg_iou, on_epoch=True, prog_bar=True)
        self.val_ious.clear()
    
    def calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Intersection over Union."""
        pred_binary = (pred > 0).float()
        intersection = (pred_binary * target).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = intersection / (union + 1e-6)
        return iou.mean()
    
    def _log_predictions(self, batch, outputs):
        """Save prediction visualizations."""
        # Implementation would save visualizations
        pass
    
    def configure_optimizers(self):
        # Only optimize unfrozen parameters
        params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams['finetune']['learning_rate'],
            weight_decay=self.hparams['finetune']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams['finetune']['num_epochs'],
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
        

class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        pred = pred.sigmoid()
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()
    
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def forward(self, pred, target):
        pred = pred.sigmoid()
        smooth = 1e-5
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
class IoULoss(nn.Module):
    """IoU loss for segmentaiton."""
    
    def forward(self, pred, target):
        pred = pred.sigmoid()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        iou = intersection / (union + 1e-6)
        return 1 - iou.mean()


def collate_fn_sam(batch):
    """Custom collate function for SAM training to handle variable prompts."""
    # Stack images and masks
    images = torch.stack([item['image'] for item in batch])
    gt_masks = torch.stack([item['gt_mask'] for item in batch])
    
    # Handle prompts - need to batch them properly
    batch_size = len(batch)
    
    # Collect all prompts
    all_points = []
    all_boxes = []
    max_points = 0
    max_boxes = 0
    
    for item in batch:
        prompts = item['prompts']
        if prompts['points'] is not None:
            coords, labels = prompts['points']
            max_points = max(max_points, len(coords))
        if prompts['boxes'] is not None:
            max_boxes = max(max_boxes, len(prompts['boxes']))
    
    # Pad and batch prompts
    batched_point_coords = []
    batched_point_labels = []
    batched_boxes = []
    
    for item in batch:
        prompts = item['prompts']
        
        # Handle points
        if prompts['points'] is not None and max_points > 0:
            coords, labels = prompts['points']
            coords = torch.tensor(coords, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32)
            # Pad if needed
            if coords.shape[0] < max_points:
                pad_size = max_points - coords.shape[0]
                coords = torch.cat([coords, torch.zeros(pad_size, 2)])
                labels = torch.cat([labels, torch.zeros(pad_size)])
            coords = coords[:1] if coords.shape[0] > 1 else torch.zeros(1, 2)
            labels = labels[:1] if labels.shape[0] > 1 else torch.zeros(1)
            batched_point_coords.append(coords)
            batched_point_labels.append(labels)
        elif max_points > 0:
            # No points for this sample, add zeros
            batched_point_coords.append(torch.zeros(max_points, 2))
            batched_point_labels.append(torch.zeros(max_points))
        print(batched_point_coords[-1].shape, batched_point_labels[-1].shape)
        
        # Handle boxes
        if prompts['boxes'] is not None and max_boxes > 0:
            boxes = torch.tensor(prompts['boxes'], dtype=torch.float32)
            if boxes.shape[0] < max_boxes:
                pad_size = max_boxes - boxes.shape[0]
                boxes = torch.cat([boxes, torch.zeros(pad_size, 4)])
            batched_boxes.append(boxes)
        elif max_boxes > 0:
            batched_boxes.append(torch.zeros(max_boxes, 4))
    
    # Create batched prompts
    batched_prompts = {}
    if batched_point_coords:
        batched_prompts['points'] = (
            torch.stack(batched_point_coords),
            torch.stack(batched_point_labels)
        )
    else:
        batched_prompts['points'] = None
    
    if batched_boxes:
        batched_prompts['boxes'] = torch.stack(batched_boxes)
    else:
        batched_prompts['boxes'] = None
    
    batched_prompts['mask_input'] = None
    
    return {
        'image': images,
        'gt_mask': gt_masks,
        'prompts': batched_prompts,
        'image_paths': [item['image_path'] for item in batch]
    }


def train_sam_model(config_path: str, output_dir: str):
    """Main training function for SAM fine-tuning."""
    
    # Load config
    config = load_config(config_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    sam_transform = ResizeLongestSide(config['data']['patch_size'])
    
    # Load manifest
    manifest_df = pd.read_csv(config['data']['manifest_path'])
    manifest_df['has_ship'] = manifest_df['EncodedPixels'].notnull().astype(int)
    manifest_df['patch_path'] = manifest_df['ImageId'].apply(
        lambda x: f"data/airbus-ship-detection/train_v2/{x}"
    )
    
    train_df = manifest_df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    val_df = manifest_df.drop(train_df.index).reset_index(drop=True)
    
    train_dataset = ShipSAMDataset(
        train_df,
        sam_transform,
        patch_size=config['data']['patch_size'],
        use_vit_prompts=config['vit_integration']['use_vit_prompts'],
    )
    
    val_dataset = ShipSAMDataset(
        val_df,
        sam_transform,
        patch_size=config['data']['patch_size'],
        use_vit_prompts=config['vit_integration']['use_vit_prompts'],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['finetune']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn_sam
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['finetune']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn_sam
    )
    
    model = SAMShipSegmentation(config)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='sam-{epoch:02d}-{val_iou:.3f}',
            monitor='val_iou',
            mode='max',
            save_top_k=3,
        ),
        EarlyStopping(
            monitor='val_iou',
            patience=5,
            mode='max'
        )
    ]
    train_logger = TensorBoardLogger(
        save_dir=output_dir,
        name='sam_logs',
    )
    trainer = pl.Trainer(
        max_epochs=config['finetune']['num_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=train_logger,
        callbacks=callbacks,
        accumulate_grad_batches=4,
    )
    
    if config['finetune']['enable']:
        logger.info("Starting SAM fine-tuning...")
        trainer.fit(model, train_loader, val_loader)
    else:
        logger.info("Fine-tuning disabled, using zero-shot SAM.")
    
    torch.save(model.sam.state_dict(), os.path.join(output_dir, 'sam_final.pth'))
    logger.info(f"Model saved to {output_dir}/sam_final.pth")