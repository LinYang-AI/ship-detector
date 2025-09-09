import os
import yaml
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLEProcessor:
    @staticmethod
    def decode_rle(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
        """Decode RLE string to binary mask.
        
        Args:
            mask_rle: Run-length encoded string
            shape: (height, width) of target image
        Returns:
            Binary mask array of shape (height, width)
        """
        if pd.isna(mask_rle) or mask_rle == '' or mask_rle == '0':
            return np.zeros(shape, dtype=np.uint8)

        s = mask_rle.split()
        starts = np.array(s[0::2], dtype=int) - 1  # RLE use 1-based indexing
        lengths = np.array(s[1::2], dtype=int)
        ends = starts + lengths
        
        # Create mask
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, end in zip(starts, ends):
            img[start:end] = 1
        
        # Important: RLE uses column-major (Fortran-style) order
        return img.reshape(shape, order='F')
    
    @staticmethod
    def encode_rle(mask: np.ndarray) -> str:
        """Encode binary mask to RLE string.
        
        Args:
            mask: Binary mask array
        
        Returns:
            RLE encoded string
        """
        # Flatten in column-major order
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        
        if len(runs) == 2 and runs[1] == len(pixels) - 1:
            return ''  # No mask
        
        return ' '.join(str(x) for x in runs)
    
    @staticmethod
    def rle_to_polygon(mask_rle: str, shape: Tuple[int, int]) -> List[List[float]]:
        """Convert RLE to polygon coordinates for YOLO format.
        
        Args:
            mask_rle: RLE encoded string
            shape: (height, width) of image
        
        Returns:
            List of normalized polygon coordinates
        """
        mask = RLEProcessor.decode_rle(mask_rle, shape)
        
        if mask.sum() == 0:
            return []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Simplify contour
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:  # Valid polygon
                # Normalize coordinates
                polygon = []
                for point in approx:
                    x = point[0][0] / shape[1]  # Normalize x
                    y = point[0][1] / shape[0]  # Normalize y
                    polygon.extend([x, y])
                
                polygons.append(polygon)
        
        return polygons
    

class YOLODatasetCreator:
    """Create YOLOv8 segmentation dataset from RLE annotations."""
    
    def __init__(self, manifest_path: str, images_dir: str, output_dir: str):
        """
        Args:
            manifest_path: Path to CSV with EncodedPixels and ImageId columns
            image_dir: Directory containing JPG images
            output_dir: Output directory for YOLO dataset
        """
        self.manifest = pd.read_csv(manifest_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.rle_processor = RLEProcessor()
        
        # Create output structure
        self.setup_directories()
        
        logger.info(f"Loaded manifest with {len(self.manifest)} entries.")
    
    def setup_directories(self):
        """Create YOLO dataset directory structure."""
        # Create directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    def process_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Process entire dataset and create YOLO format files."""
        
        # Group by image (multiple ships per image)
        grouped = self.manifest.groupby('ImageId')
        
        # Split dataset
        image_ids = list(grouped.groups.keys())
        np.random.shuffle(image_ids)
        
        n_train = int(len(image_ids) * train_ratio)
        n_val = int(len(image_ids) * val_ratio)
        
        train_ids = image_ids[:n_train]
        val_ids = image_ids[n_train:n_train + n_val]
        test_ids = image_ids[n_train + n_val:]
        
        logger.info(f"Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        
        # Process each split
        self._process_split(train_ids, grouped, 'train')
        self._process_split(val_ids, grouped, 'val')
        self._process_split(test_ids, grouped, 'test')
        
        # Create YAML configuration
        self._create_yaml_config()
    
    def _process_split(self, image_ids: List[str], grouped, split: str):
        """Process a data split."""
        logger.info(f"Processing {split} split...")
        
        for image_id in tqdm(image_ids, desc=f"Processing {split}"):
            # Get all RLE masks for this iamge
            image_data = grouped.get_group(image_id)
            
            # Load image to get shape
            image_path = self.images_dir / image_id
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Copy image to dataset
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
        
            height, width = image.shape[:2]
            
            # Save image to output directory
            output_image_path = self.output_dir / 'images' / split / image_id
            cv2.imwrite(str(output_image_path), image)
            
            # Process all ships in this image
            labels = []
            for _, row in image_data.iterrows():
                if pd.notna(row['EncodedPixels']) and row['EncodedPixels'] != '':
                    # Convert RLE to polygons
                    polygons = self.rle_processor.rle_to_polygon(
                        row['EncodedPixels'],
                        (height, width)
                    )
                    
                    # Add to labels (class 0 for ships)
                    for polygon in polygons:
                        if len(polygon) >= 6:
                            label = [0] + polygon  # Class ID + polygon coords
                            labels.append(label)
            
            # Save labels file
            if labels:
                label_path = self.output_dir / 'labels' / split / f"{image_id.rsplit('.', 1)[0]}.txt"
                with open(label_path, 'w') as f:
                    for label in labels:
                        f.write(' '.join(map(str, label)) + '\n')
                        
    
    def _create_yaml_config(self):
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'ship'
            },
            'nc': 1  # Number of classes
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Created dataset configuration: {yaml_path}")
        
    def verify_dataset(self, num_samples: int = 5):
        """Verify dataset by visualizing some samples."""
        logger.info("Verifying dataset...")
        
        vis_dir = self.output_dir / 'verification'
        vis_dir.mkdir(exist_ok=True)
        
        # Check train split
        train_images = list((self.output_dir / 'images' / 'train').glob('*.jpg'))[:num_samples]
        
        for img_path in train_images:
            # Load image
            image = cv2.imread(str(img_path))
            overlay = image.copy()
            
            # Load labels
            label_path = self.output_dir / 'labels' / 'train' / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        class_id = int(parts[0])
                        polygon = parts[1:]
                        
                        # Convert normalized to pixel coordinates
                        h, w = image.shape[:2]
                        points = []
                        for i in range(0, len(polygon), 2):
                            x = int(polygon[i] * w)
                            y = int(polygon[i+1] * h)
                            points.append([x, y])
                        
                        points = np.array(points, np.int32)
                        
                        # Draw polygon
                        cv2.fillPoly(overlay, [points], (0, 255, 0))
                        cv2.polylines(image, [points], True, (0, 0, 255), 2)
            
            # Blen overlay
            result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            
            # Save visualization
            vis_path = vis_dir / f"verify_{img_path.name}"
            cv2.imwrite(str(vis_path), result)
        
        logger.info(f"Saved verification iamges to {vis_dir}")
        
    def create_synthetic_data(output_dir: str, num_images: int = 10):
        """Create synthetic dataset for testing."""
        
        output_dir = Path(output_dir)
        
        # Create directories
        images_dir = output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create manifest
        manifest_data = []
        rle_processor = RLEProcessor()
        
        for i in range(num_images):
            # Create synthetic image (ocean with ships):
            image = np.ones((768, 768, 3), dtype=np.uint8) * 50  # Dark blue ocean
            mask = np.zeros((768, 768), dtype=np.uint8)
            
            # Add random ships
            num_ships = np.random.randint(1, 5)
            for j in range(num_ships):
                # Random ship position and size
                x = np.random.randint(50, 668)
                y = np.random.randint(50, 668)
                w = np.random.randint(30, 100)
                h = np.random.randint(15, 50)
                
                angle = np.random.randint(0, 180)
                
                # Draw ship (rotated rectangle)
                center = (x + w//2, y + h//2)
                rect = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Create ship shape
                cv2.ellipse(image, center, (w//2, h//2), angle, 0, 360, (200, 200, 200), -1)
                cv2.ellipse(mask, center, (w//2, h//2), angle, 0, 360, 255, -1)
                
            # Save image
            image_name = f"synthetic_{i:04d}.jpg"
            cv2.imwrite(str(images_dir / image_name), image)
            
            # Encode mask to RLE
            rle = rle_processor.encode_rle(mask)
            
            manifest_data.append({
                'ImageId': image_name,
                'EncodedPixels': rle
            })
            
            # Save manifest
            manifest_path = output_dir / 'train_synthetic.csv'
            pd.DataFrame(manifest_data).to_csv(manifest_path, index=False)
            
