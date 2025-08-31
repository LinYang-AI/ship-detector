import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from ship_detector.scripts.train_vit import *
from ship_detector.scripts.utils import load_config


def training_classifier():
    pl.seed_everything(42)
    
    config = load_config("configs/vit.yaml")
    manifest_path = "data/airbus-ship-detection/train_ship_segmentations_v2.csv"
    output_dir = "outputs"
    Path.mkdir(output_dir, exist_ok=True)
    
    train_dataloader, val_dataloader = create_data_loader(manifest_path, config)
    
    if "vit" in config['model']['name']:
        pre_name = 'vit'
        model = ViTShipClassifier(config)
        # Continue with training setup...
    else:
        pre_name = 'resnet'
        model = ResNetShipClassifier(config)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename=pre_name + '-{epoch:02d}-{val_acc:.3f}',
            monitor='val_loss',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            mode='min',
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=pre_name + '_logs',
    )
    
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
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
    save_path = os.path.join(output_dir, f'{pre_name}_ship_final.pth')
    torch.save(model.state_dict(), save_path)

