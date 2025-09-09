import os
import gc
import yaml
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
import psutil
import threading
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
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

warnings.filterwarnings('ignore')


# ============================================================================
# LoRA Implementation for ViT
# ============================================================================


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for parameter-efficient fine-tuning."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation."""
        lora_output = x @ self.lora_A @ self.lora_B
        return self.dropout(lora_output) * self.scaling


def add_lora_to_vit(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    target_modules: List[str] = ['qkv', 'proj', 'fc1', 'fc2']
) -> nn.Module:
    """Add LoRA layers to Vision Transformer.
    
    Args:
        model: ViT model
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: Which modules to add LoRA to
    
    Returns:
        Model with LoRA layers added
    """
    lora_layers = {}
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Store original weight
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                
                # Add LoRA layer
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha
                )
                safe_layer_name = name.replace('.', '_')
                lora_layers[safe_layer_name] = lora_layer
    
    # Monkey-patch forward methods to include LoRA
    def make_forward_with_lora(original_module, lora_layer):
        def forward_with_lora(x):
            original_output = original_module(x)
            lora_output = lora_layer(x)
            return original_output + lora_output
        return forward_with_lora
    
    for name, module in model.named_modules():
        if name in lora_layers:
            module.forward = make_forward_with_lora(module, lora_layers[name])
    
    # Store LoRA layers in model
    model.lora_layers = nn.ModuleDict(lora_layers)
    
    return model


# ============================================================================
# Memory-Efficient Dataset
# ============================================================================

class LazyLoadDataset(Dataset):
    """Dataset that loads images on-demand to save RAM."""
    
    def __init__(
        self,
        config: Dict,
        manifest_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        cache_size: int = 0,  # Number of images to cache in memory
        preload_batch: bool = False
    ):
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.cache_size = cache_size
        self.preload_batch = preload_batch
        self.config = config
        self.preprocessing_method = config['data'].get(
            'preprocessing_method', 'adaptive'
        )
        
        # LRU cache for recently accessed images
        if cache_size > 0:
            from collections import OrderedDict
            self.cache = OrderedDict()
        else:
            self.cache = None
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image from disk."""
        try:
            # Use PIL for lower memory footprint
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                # Convert to numpy array
                return np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check cache first
        if self.cache is not None and idx in self.cache:
            # Move to end (most recently used)
            image = self.cache.pop(idx)
            self.cache[idx] = image
        else:
            # Load from disk
            row = self.manifest.iloc[idx]
            img_path = row['patch_path']
            image = self._load_image(img_path)
            
            # Add to cache if enabled
            if self.cache is not None:
                # Remove oldest if cache is full
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                self.cache[idx] = image
        
        # Get label
        row = self.manifest.iloc[idx]
        label = torch.tensor(row['has_ship'], dtype=torch.float32)
        image = self.preprocess_image(image)
        # Apply transforms
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def preprocess_image(self, image):
        """Apply ship-preserving preprocessing"""
        if self.preprocessing_method == 'multiscale':
            return self.multiscale_resize_with_context(image)
        elif self.preprocessing_method == 'adaptive':
            return  self.adaptive_resize_preserve_ships(image)
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
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        if self.cache is not None:
            self.cache.clear()
        gc.collect()


class StreamingDataset(IterableDataset):
    """Streaming dataset for extremely large datasets."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        manifest_path: str,
        transform: Optional[transforms.Compose] = None,
        chunk_size: int = 1000,
        shuffle_buffer: int = 100
    ):
        self.manifest_path = manifest_path
        self.transform = transform
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer
        self.preprocessing_method = config['data'].get(
            'preprocessing_method', 'adaptive'
        )
    
    def __iter__(self):
        # Read manifest in chunks
        for chunk_df in pd.read_csv(self.manifest_path, chunksize=self.chunk_size):
            # Shuffle within chunk
            chunk_df = chunk_df.sample(frac=1).reset_index(drop=True)
            
            for _, row in chunk_df.iterrows():
                # Load image
                try:
                    image = Image.imread(row['patch_path'])
                    if image is None:
                        raise FileNotFoundError(f"Image not found: {row['patch_path']}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image.shape[:2] != (768, 768):
                        image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_AREA)
                    image = self.preprocess_image(image)
                    image = Image.fromarray(image)
                    
                    if self.transform:
                        if isinstance(self.transform, list):
                            image = transforms.Compose(self.transform)(image)
                        else:
                            image = self.transform(image)
                    label = torch.tensor(row['has_ship'], dtype=torch.float32)

                    yield image, label

                    # if self.transform:
                    #     img = self.transform(img)
                    # else:
                    #     img = transforms.ToTensor()(img)
                    
                    # label = torch.tensor(row['has_ship'], dtype=torch.float32)
                    
                    # yield img, label
                    
                except Exception as e:
                    print(f"Error loading {row['patch_path']}: {e}")
                    continue
    
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
        # transforms.COmpose or signle callable
        if self.transform:
            if isinstance(self.transform, list):
                
                # If mistakenly passed a list, compose it
                image = transforms.Compose(self.transform)(image)
            else:
                image = self.transform(image)
        label = torch.tensor(row['has_ship'], dtype=torch.float32)
        return image, label
    
    def preprocess_image(self, image):
        """Apply ship-preserving preprocessing"""
        if self.preprocessing_method == 'multiscale':
            return self.multiscale_resize_with_context(image)
        elif self.preprocessing_method == 'adaptive':
            return  self.adaptive_resize_preserve_ships(image)
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


# ============================================================================
# Memory-Efficient ViT with LoRA
# ============================================================================

class LoRALinear(nn.Module):
    """Proper LoRA Linear layer implementation"""
    
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA parameters - these MUST have requires_grad=True
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
        # Ensure gradients are enabled for LoRA parameters
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        
    def forward(self, x):
        # Original forward pass (frozen)
        original_out = self.original_layer(x)
        
        # LoRA adaptation
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        
        # Combine with scaling
        return original_out + (self.alpha / self.rank) * lora_out

class LoRAAttention(nn.Module):
    """LoRA adapter for attention layers"""
    
    def __init__(self, original_attention, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.original_attention = original_attention
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original attention
        for param in self.original_attention.parameters():
            param.requires_grad = False
        
        # Add LoRA to query, key, value projections
        if hasattr(original_attention, 'qkv'):
            self.qkv_lora = LoRALinear(original_attention.qkv, rank, alpha)
        
        if hasattr(original_attention, 'proj'):
            self.proj_lora = LoRALinear(original_attention.proj, rank, alpha)
    
    def forward(self, x, *args, **kwargs):
        # This is a simplified version - you'd need to adapt based on the specific attention implementation
        return self.original_attention(x, *args, **kwargs)


class EfficientViTClassifier(pl.LightningModule):
    """Fixed ViT classifier with proper LoRA implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            config['model']['name'],
            pretrained=config['model']['pretrained'],
            num_classes=0,  # Remove original head
        )
        
        # Add LoRA adapters
        self.add_lora_to_vit(
            rank=config['model'].get('lora_rank', 16),
            alpha=config['model'].get('lora_alpha', 32)
        )
        
        # Binary classification head (always trainable)
        self.classifier = nn.Sequential(
            nn.Dropout(config['model'].get('dropout', 0.1)),
            nn.Linear(self.backbone.num_features, 1)
        )
        
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        # Loss function
        pos_weight = torch.tensor([config['training'].get('pos_weight', 1.0)])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def add_lora_to_vit(self, rank: int = 16, alpha: float = 32):
        """Add LoRA adapters to ViT transformer blocks"""
        
        # First, freeze ALL original parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Add LoRA to transformer blocks
        for name, module in self.backbone.named_modules():
            if 'blocks' in name and 'attn.qkv' in name:
                # Replace QKV projection with LoRA version
                parent_name = name.rsplit('.', 1)[0]
                parent_module = dict(self.backbone.named_modules())[parent_name]
                
                # Get the qkv layer
                qkv_layer = module
                lora_qkv = LoRALinear(qkv_layer, rank, alpha)
                
                # Replace the layer
                setattr(parent_module, 'qkv', lora_qkv)
                
            elif 'blocks' in name and 'attn.proj' in name and 'drop' not in name:
                # Replace attention projection with LoRA version
                parent_name = name.rsplit('.', 1)[0]
                parent_module = dict(self.backbone.named_modules())[parent_name]
                
                proj_layer = module
                lora_proj = LoRALinear(proj_layer, rank, alpha)
                setattr(parent_module, 'proj', lora_proj)
                
            elif 'blocks' in name and 'mlp.fc1' in name:
                # Replace MLP layers with LoRA versions
                parent_name = name.rsplit('.', 1)[0]
                parent_module = dict(self.backbone.named_modules())[parent_name]
                
                fc1_layer = module
                lora_fc1 = LoRALinear(fc1_layer, rank, alpha)
                setattr(parent_module, 'fc1', lora_fc1)
                
            elif 'blocks' in name and 'mlp.fc2' in name:
                parent_name = name.rsplit('.', 1)[0]
                parent_module = dict(self.backbone.named_modules())[parent_name]
                
                fc2_layer = module
                lora_fc2 = LoRALinear(fc2_layer, rank, alpha)
                setattr(parent_module, 'fc2', lora_fc2)
        
        # Verify we have trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"LoRA Setup Complete:")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params:.4f}")
        
        if trainable_params == 0:
            raise RuntimeError("No trainable parameters! LoRA setup failed.")
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)  # Shape: (batch_size, num_features)
        
        # Binary classification
        logits = self.classifier(features)  # Shape: (batch_size, 1)
        
        return logits.squeeze(-1)  # Shape: (batch_size,)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # CRITICAL FIX: Verify loss requires grad
        if not loss.requires_grad:
            raise RuntimeError(
                f"Loss does not require grad! "
                f"Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
            )
        
        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {
            'loss': loss,
            'preds': preds,
            'labels': labels,
            'probs': probs
        }
    
    def configure_optimizers(self):
        # Only optimize trainable parameters (LoRA + classifier)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found!")
        
        opt_config = self.hparams['optimizer']
        
        if opt_config['name'] == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        elif opt_config['name'] == 'adam':
            optimizer = torch.optim.Adam(
                trainable_params,
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
        else:
            return optimizer


# ============================================================================
# Memory Monitoring
# ============================================================================

class MemoryMonitor:
    """Monitor and manage system memory usage."""
    
    def __init__(self, threshold_percent: float = 80.0):
        self.threshold_percent = threshold_percent
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            gpu_memory = {
                'gpu_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'gpu_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'gpu_free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                               torch.cuda.memory_allocated()) / 1e9
            }
        
        return {
            'ram_used_gb': memory.used / 1e9,
            'ram_available_gb': memory.available / 1e9,
            'ram_percent': memory.percent,
            'process_ram_gb': self.process.memory_info().rss / 1e9,
            **gpu_memory
        }
    
    def check_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        memory = psutil.virtual_memory()
        return memory.percent > self.threshold_percent
    
    def free_memory(self):
        """Attempt to free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ============================================================================
# Data Loading Utilities
# ============================================================================

def get_augmentation_transforms(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    aug_config = config['augmentation']
    
    # Training transforms
    train_transforms = []
    
    # Geometric augmentations (ship-aware)
    train_transforms.extend([
        transforms.RandomHorizontalFlip(p=aug_config.get('hflip_prob', 0.5)),
        transforms.RandomVerticalFlip(p=aug_config.get('vflip_prob', 0.5)),
    ])
    
    if aug_config.get('rotation', False):
        train_transforms.append(
            transforms.RandomRotation(degrees=0.5)
        )
    
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
    
    if aug_config.get('gaussian_blur_prob', 0) > 0:
        train_transforms.append(
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=aug_config['gaussian_blur_prob'])
        )
    
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

def create_efficient_data_loaders(
    manifest_path: str,
    config: Dict[str, Any],
    memory_monitor: Optional[MemoryMonitor] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create memory-efficient data loaders.
    
    Args:
        manifest_path: Path to data manifest
        config: Configuration dictionary
        memory_monitor: Optional memory monitor
    
    Returns:
        Training and validation data loaders
    """
    # Load manifest
    df = pd.read_csv(manifest_path)
    df['has_ship'] = df['EncodedPixels'].notnull().astype(int)
    df['patch_path'] = df['ImageId'].apply(lambda x: f"{config['data']['train']}/{x}")
    # # Randomly select half
    # df, _ = train_test_split(
    #     df,
    #     test_size=0.5,
    #     random_state=config['data']['random_seed'],
    #     stratify=df['has_ship']
    # )
    
    # Check memory before loading
    if memory_monitor and memory_monitor.check_memory_critical():
        print("Warning: High memory usage detected. Using streaming dataset.")
        use_streaming = True
    else:
        use_streaming = config['data'].get('use_streaming', False)
    
    # Split data
    train_df, val_df = train_test_split(
        df,
        test_size=config['data']['val_split'],
        random_state=config['data']['random_seed'],
        stratify=df['has_ship']
    )
    
    print(f"Training samples: {len(train_df)} (Ships: {train_df['has_ship'].sum()})")
    print(f"Validation samples: {len(val_df)} (Ships: {val_df['has_ship'].sum()})")
    
    train_transform, val_transform = get_augmentation_transforms(config)
    
    # Create datasets
    if use_streaming:
        # Save temporary manifests
        train_manifest_path = 'temp_train_manifest.csv'
        val_manifest_path = 'temp_val_manifest.csv'
        train_df.to_csv(train_manifest_path, index=False)
        val_df.to_csv(val_manifest_path, index=False)
        
        train_dataset = StreamingDataset(
            config,
            train_manifest_path,
            transform=train_transform,
            chunk_size=config['data'].get('chunk_size', 1000)
        )
        val_dataset = StreamingDataset(
            config,
            val_manifest_path,
            transform=val_transform,
            chunk_size=config['data'].get('chunk_size', 1000)
        )
    else:
        train_dataset = LazyLoadDataset(
            config,
            train_df,
            transform=train_transform,
            cache_size=config['data'].get('cache_size', 100)
        )
        val_dataset = LazyLoadDataset(
            config,
            val_df,
            transform=val_transform,
            cache_size=config['data'].get('cache_size', 100)
        )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=not use_streaming,  # Streaming dataset handles its own shuffling
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True),
        persistent_workers=config['data'].get('persistent_workers', False),
        prefetch_factor=config['data'].get('prefetch_factor', 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True),
        persistent_workers=config['data'].get('persistent_workers', False)
    )
    
    return train_loader, val_loader


# ============================================================================
# Main Training Function
# ============================================================================

def main(config_path: str, manifest_path: str, output_dir: str):
    """Main training function with memory optimization and LoRA."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set memory-efficient options
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(threshold_percent=80.0)
    
    # Log initial memory
    print("\nInitial Memory Status:")
    for key, value in memory_monitor.get_memory_usage().items():
        print(f"  {key}: {value:.2f}")
    
    # Set seed
    pl.seed_everything(config['data']['random_seed'])
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_efficient_data_loaders(
        manifest_path,
        config,
        memory_monitor
    )
    
    # Initialize model
    model = EfficientViTClassifier(config)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='vit-{epoch:02d}-{val_acc:.3f}',
            monitor='val_acc',
            mode='max',
            save_top_k=2,  # Save less checkpoints to save disk space
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Trainer with memory-efficient settings
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        precision=config['training'].get('precision', 16),
        log_every_n_steps=10,
        val_check_interval=config['training'].get('val_check_interval', 1.0),
        limit_train_batches=config['training'].get('limit_train_batches', 1.0),
        limit_val_batches=config['training'].get('limit_val_batches', 1.0)
    )
    
    # Train model
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Final memory status
    print("\nFinal Memory Status:")
    for key, value in memory_monitor.get_memory_usage().items():
        print(f"  {key}: {value:.2f}")
    
    # Save final model
    if config['model'].get('use_lora', False):
        # Save only LoRA weights
        lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k}
        torch.save(lora_state_dict, os.path.join(output_dir, 'lora_weights.pt'))
        print(f"LoRA weights saved to {output_dir}/lora_weights.pt")
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, 'vit_final.pt'))
    
    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient ViT training with LoRA")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--manifest', type=str, required=True, help='Path to data manifest')
    parser.add_argument('--output-dir', type=str, default='./models/vit_efficient', help='Output directory')
    
    args = parser.parse_args()
    main(args.config, args.manifest, args.output_dir)