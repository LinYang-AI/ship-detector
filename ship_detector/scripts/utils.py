import argparse
import yaml
from typing import Dict, Any
import cv2
import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_args():
    parser = argparse.ArgumentParser(description="Ship Detector")
    parser.add_argument("-train_vit", action="store_true", help="Train ViT model")
    parser.add_argument("-train_unet", action="store_true", help="Train U-Net model")
    parser.add_argument("-train_sam", action="store_true", help="Train SAM model")

    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )

    return parser.parse_args()


def get_task(args):
    if args.train_vit:
        return "train_vit"
    elif args.train_unet:
        return "train_unet"
    elif args.train_sam:
        return "train_sam"
    else:
        raise ValueError("No valid task specified. Use -train_vit or -train_unet.")


def rle_decode(rle, shape):
    """
    rle: rle字符串
    shape: (height, width)
    """
    import numpy as np

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if pd.isna(rle):
        return mask.reshape(shape)
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    for start, length in zip(starts, lengths):
        mask[start : start + length] = 1
    return mask.reshape(shape, order="F")  # 注意order='F'（列优先）
