import os
from typing import Literal
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from ultralytics import YOLO
from ship_detector.scripts.train_vit import create_vit_data_loader, ViTShipClassifier
from ship_detector.scripts.train_unet import create_unet_data_loaders, UNetShipSegmentation
from ship_detector.scripts.train_sam import SAMShipSegmentation, ShipSAMDataset, collate_fn_sam
from ship_detector.scripts.train_yolo_seg import YOLOShipSegmentation, visualize_predictions
from ship_detector.scripts.utils import load_config

from segment_anything.utils.transforms import ResizeLongestSide


def train_vit_model(config_path: str, output_dir: str,):
    pl.seed_everything(42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Load config
    config = load_config(config_path)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    train_loader, val_loader = create_vit_data_loader(config=config)
    
    vit_model = ViTShipClassifier(config=config)
    
    vit_checkpoint = config['model'].get('checkpoint_path', None)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'vit/checkpoints'),
            filename='vit-{epoch:02d}-{val_acc:.3f}',
            monitor='val_loss',
            mode='max',
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('early_stopping_patience', 10),
            mode='min',
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'vit'),
        name='vit_logs',
    )
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision=config['training'].get('precision', 32),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        deterministic=True,
    )
    
    print(f"Run `tensorboard --logdir {os.path.join(output_dir, 'vit/vit_logs')}` to monitor training")
    print(f"Output directory: {output_dir}")
    if vit_checkpoint is not None and os.path.isfile(vit_checkpoint):
        print(f"Resuming from checkpoint: {vit_checkpoint}")
        trainer.fit(vit_model, train_loader, val_loader, ckpt_path=vit_checkpoint)
    else:
        print(f"\nStarting training...")
        trainer.fit(vit_model, train_loader, val_loader)
    
    final_path = os.path.join(output_dir, 'vit_ship_final.pt')
    torch.save(vit_model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")


def train_unet_model(config_path: str, output_dir: str):
    """Main training function."""
    
    # Load config
    config = load_config(config_path)
    
    # Set seeds
    pl.seed_everything(config['data']['random_seed'])
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_unet_data_loaders(config)

    # Initialize model
    model = UNetShipSegmentation(config)
    
    unet_checkpoint = config['model'].get('checkpoint_path', None)
    
    # Print model info
    print(f"\nModel: U-Net with {config['model']['encoder']} encoder")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'unet/checkpoints'),
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
        save_dir=os.path.join(output_dir, 'unet'),
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
    print(f"Run `tensorboard --logdir {os.path.join(output_dir, 'unet/unet_logs')}` to monitor training")
    print(f"Output directory: {output_dir}")
    if unet_checkpoint is not None and os.path.isfile(unet_checkpoint):
        print(f"Resuming from checkpoint: {unet_checkpoint}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=unet_checkpoint)
    else:
        print(f"\nStarting training...")
        trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = os.path.join(output_dir, 'unet_ship_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")
    
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best validation IoU: {callbacks[0].best_model_score:.4f}")


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
        trainer.fit(model, train_loader, val_loader)
    else:
        print("Fine-tuning disabled in config. Exiting.")
        return
    
    torch.save(model.sam.state_dict(), os.path.join(output_dir, 'sam_final.pth'))
    print(f"Model saved to {output_dir}/sam_final.pth")
    
    
def train_yolov8_model(
    config_path: str, 
    data_path: str, 
    mode: Literal['train', 'val', 'predict', 'export'] = 'train', 
    weights_path: str | None = None, 
    img_source: str | None = None, 
    resume: bool = False,
    checkpoint_path: str | None = None,
    export_format: Literal['onnx', 'torchscript', 'coreml', 'tflite'] = 'torchscript'):
    
    yolo = YOLOShipSegmentation(config_path)
    if checkpoint_path:
        yolo.model = YOLO(checkpoint_path)
    if mode == 'train':
        yolo.train(data_path, resume)
    elif mode == 'val':
        yolo.validate(data_path, weights_path)
    elif mode == 'predict':
        if not img_source:
            raise ValueError("'img_source' required for prediction")
        if weights_path:
            yolo.model = YOLO(weights_path)
        
        results = yolo.predict(img_source)
        
        # Visualize results
        output_path = f"prediction_{Path(img_source).stem}.jpg"
        visualize_predictions(img_source, results, output_path)
    
    elif mode == 'export':
        # Export model
        if weights_path:
            yolo.model = YOLO(weights_path)
        yolo.export_model(export_format)
