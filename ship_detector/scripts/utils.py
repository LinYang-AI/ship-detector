import argparse
import yaml
from typing import Dict, Any


def load_config(config_path: str) ->  Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_args():
    parser = argparse.ArgumentParser(description="Ship Detector")
    parser.add_argument('-train_vit', action='store_true', help='Train ViT model')
    parser.add_argument('-train_unet', action='store_true', help='Train U-Net model')
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    
    return parser.parse_args()

def get_task(args):
    if args.train_vit:
        return 'train_vit'
    elif args.train_unet:
        return 'train_unet'
    else:
        raise ValueError("No valid task specified. Use -train_vit or -train_unet.")