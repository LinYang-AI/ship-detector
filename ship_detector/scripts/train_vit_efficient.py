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
        manifest_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        cache_size: int = 0,  # Number of images to cache in memory
        preload_batch: bool = False
    ):
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.cache_size = cache_size
        self.preload_batch = preload_batch
        
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
        
        # Apply transforms
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        if self.cache is not None:
            self.cache.clear()
        gc.collect()


class StreamingDataset(IterableDataset):
    """Streaming dataset for extremely large datasets."""
    
    def __init__(
        self,
        manifest_path: str,
        transform: Optional[transforms.Compose] = None,
        chunk_size: int = 1000,
        shuffle_buffer: int = 100
    ):
        self.manifest_path = manifest_path
        self.transform = transform
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer
    
    def __iter__(self):
        # Read manifest in chunks
        for chunk_df in pd.read_csv(self.manifest_path, chunksize=self.chunk_size):
            # Shuffle within chunk
            chunk_df = chunk_df.sample(frac=1).reset_index(drop=True)
            
            for _, row in chunk_df.iterrows():
                # Load image
                try:
                    img = Image.open(row['patch_path']).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    else:
                        img = transforms.ToTensor()(img)
                    
                    label = torch.tensor(row['has_ship'], dtype=torch.float32)
                    
                    yield img, label
                    
                except Exception as e:
                    print(f"Error loading {row['patch_path']}: {e}")
                    continue


# ============================================================================
# Memory-Efficient ViT with LoRA
# ============================================================================

class EfficientViTClassifier(pl.LightningModule):
    """Memory-efficient ViT with LoRA for ship classification."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Load pretrained ViT
        self.model = timm.create_model(
            config['model']['name'],
            pretrained=config['model']['pretrained'],
            num_classes=1
        )
        
        # Apply LoRA if enabled
        if config['model'].get('use_lora', False):
            self.model = add_lora_to_vit(
                self.model,
                rank=config['model'].get('lora_rank', 16),
                alpha=config['model'].get('lora_alpha', 16.0),
                target_modules=config['model'].get('lora_target_modules', ['qkv', 'proj'])
            )
            self._freeze_base_model()
            print(f"LoRA enabled with rank={config['model'].get('lora_rank', 16)}")
        
        # Loss function
        pos_weight = torch.tensor([config['training'].get('pos_weight', 1.0)])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Gradient checkpointing for memory efficiency
        if config['model'].get('gradient_checkpointing', False):
            self.model.set_grad_checkpointing(enable=True)
        
        # Mixed precision
        self.automatic_optimization = True
        
        # Log model stats
        self._log_model_stats()
    
    def _freeze_base_model(self):
        """Freeze base model parameters when using LoRA."""
        for name, param in self.model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    
    def _log_model_stats(self):
        """Log model parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("="*50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Reduction: {(1 - trainable_params/total_params)*100:.1f}%")
        print("="*50)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass with gradient accumulation
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.sigmoid(outputs) > 0.5
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Free up memory periodically
        if batch_idx % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels)
        
        probs = torch.sigmoid(outputs)
        preds = probs > 0.5
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'acc': acc}
    
    def configure_optimizers(self):
        # Only optimize LoRA parameters if enabled
        if self.hparams['model'].get('use_lora', False):
            # Only LoRA parameters
            lora_params = [p for n, p in self.named_parameters() if 'lora' in n and p.requires_grad]
            params = lora_params
        else:
            # All parameters
            params = self.parameters()
        
        # Optimizer
        opt_config = self.hparams['optimizer']
        if opt_config['name'] == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['name'] == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        elif opt_config['name'] == '8bit_adam':
            # Use 8-bit Adam for memory efficiency
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                params,
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        
        # Scheduler
        scheduler_config = self.hparams['scheduler']
        if scheduler_config['name'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config['T_max']
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        
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
    df['patch_path'] = df['ImageId'].apply(lambda x: os.path.join(config['data']['train'], x))
    
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
    
    # Create transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    if use_streaming:
        # Save temporary manifests
        train_manifest_path = 'temp_train_manifest.csv'
        val_manifest_path = 'temp_val_manifest.csv'
        train_df.to_csv(train_manifest_path, index=False)
        val_df.to_csv(val_manifest_path, index=False)
        
        train_dataset = StreamingDataset(
            train_manifest_path,
            transform=train_transform,
            chunk_size=config['data'].get('chunk_size', 1000)
        )
        val_dataset = StreamingDataset(
            val_manifest_path,
            transform=val_transform,
            chunk_size=config['data'].get('chunk_size', 1000)
        )
    else:
        train_dataset = LazyLoadDataset(
            train_df,
            transform=train_transform,
            cache_size=config['data'].get('cache_size', 100)
        )
        val_dataset = LazyLoadDataset(
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