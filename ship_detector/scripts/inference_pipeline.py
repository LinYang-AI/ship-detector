import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import geojson

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_vit import ViTShipClassifier
from scripts.train_unet import UNetShipSegmentation
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline."""
    vit_checkpoint: str
    vit_config: str
    unet_checkpoint: str
    unet_config: str
    patch_size: int = 224
    overlap: int = 32
    batch_size: int = 32
    confidence_threshold: float = 0.5
    min_ship_size: int = 10
    device: str = 'cuda'
    use_s3: bool = False
    s3_bucket: Optional[str] = None
    output_format: List[str] = None  # ['mask', 'json', 'overlay', 'crops']
    save_patch_predictions: bool = False
    
    def __post_init__(self):
        if self.output_format is None:
            self.output_format = ['mask', 'json', 'overlay']


class ImageDataset(Dataset):
    """Dataset for image patches."""
    
    def __init__(self, patches: List[Dict], transform=None):
        self.patches = patches
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        image = patch_info['image']
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)['image']
            else:
                # Convert numpy array to PIL Image for torchvision transforms
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype(np.uint8))
                image = self.transform(image)
        
        return image, idx


class InferencePipeline:
    """Main inference pipeline for JPG/PNG images."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Load models
        logger.info("Loading models...")
        self.vit_model = self._load_vit_model()
        self.unet_model = self._load_unet_model()
        
        # Setup transforms
        self.vit_transform = self._get_vit_transform()
        self.unet_transform = self._get_unet_transform()
        
        # S3 client if needed
        if config.use_s3:
            import boto3
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None
        
        logger.info(f"Pipeline initialized. Device: {self.device}")
    
    def _load_vit_model(self):
        """Load ViT classifier model."""
        import yaml
        with open(self.config.vit_config, 'r') as f:
            vit_config = yaml.safe_load(f)
        
        model = ViTShipClassifier(vit_config)
        
        if self.config.vit_checkpoint.endswith('.ckpt'):
            checkpoint = torch.load(self.config.vit_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(torch.load(self.config.vit_checkpoint, map_location='cpu', weights_only=False))
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_unet_model(self):
        """Load U-Net segmentation model."""
        import yaml
        with open(self.config.unet_config, 'r') as f:
            unet_config = yaml.safe_load(f)
        
        model = UNetShipSegmentation(unet_config)
        
        if self.config.unet_checkpoint.endswith('.ckpt'):
            checkpoint = torch.load(self.config.unet_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(torch.load(self.config.unet_checkpoint, map_location='cpu'))
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_vit_transform(self):
        """Get ViT preprocessing transform."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    def _get_unet_transform(self):
        """Get U-Net preprocessing transform."""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path.
        
        Args:
            image_path: Path to JPG/PNG image
        
        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image based on extension
        ext = Path(image_path).suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            # Use PIL for better compatibility
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            # Fallback to OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Loaded image: {image_path} - Shape: {image.shape}")
        return image
    
    def tile_image(self, image: np.ndarray) -> Tuple[List[Dict], Dict]:
        """Tile image into patches.
        
        Args:
            image: Input image as numpy array (H, W, C)
        
        Returns:
            Tuple of (patches_list, metadata)
        """
        patches = []
        h, w = image.shape[:2]
        
        # Store metadata
        metadata = {
            'width': w,
            'height': h,
            'channels': image.shape[2] if len(image.shape) > 2 else 1,
            'dtype': str(image.dtype)
        }
        
        # Calculate tiling
        stride = self.config.patch_size - self.config.overlap
        patch_id = 0
        
        logger.info(f"Tiling image: {w}x{h} into {self.config.patch_size}x{self.config.patch_size} patches")
        
        # Generate patches with stride
        for row in range(0, h - self.config.patch_size + 1, stride):
            for col in range(0, w - self.config.patch_size + 1, stride):
                patch_data = image[row:row+self.config.patch_size, 
                                  col:col+self.config.patch_size]
                
                # Skip uniform patches (likely background)
                if patch_data.std() < 1.0:  # Very low variance threshold
                    continue
                
                patches.append({
                    'id': patch_id,
                    'image': patch_data.copy(),
                    'row': row,
                    'col': col,
                    'coords': (row, col, row+self.config.patch_size, col+self.config.patch_size)
                })
                patch_id += 1
        
        # Handle edge cases - add patches at image borders if needed
        # Right edge
        if w % stride != 0 and w > self.config.patch_size:
            col = w - self.config.patch_size
            for row in range(0, h - self.config.patch_size + 1, stride):
                patch_data = image[row:row+self.config.patch_size,
                                  col:col+self.config.patch_size]
                
                if patch_data.std() >= 1.0:
                    patches.append({
                        'id': patch_id,
                        'image': patch_data.copy(),
                        'row': row,
                        'col': col,
                        'coords': (row, col, row+self.config.patch_size, col+self.config.patch_size)
                    })
                    patch_id += 1
        
        # Bottom edge
        if h % stride != 0 and h > self.config.patch_size:
            row = h - self.config.patch_size
            for col in range(0, w - self.config.patch_size + 1, stride):
                patch_data = image[row:row+self.config.patch_size,
                                  col:col+self.config.patch_size]
                
                if patch_data.std() >= 1.0:
                    patches.append({
                        'id': patch_id,
                        'image': patch_data.copy(),
                        'row': row,
                        'col': col,
                        'coords': (row, col, row+self.config.patch_size, col+self.config.patch_size)
                    })
                    patch_id += 1
        
        # Bottom-right corner
        if w % stride != 0 and h % stride != 0 and w > self.config.patch_size and h > self.config.patch_size:
            row = h - self.config.patch_size
            col = w - self.config.patch_size
            patch_data = image[row:row+self.config.patch_size,
                              col:col+self.config.patch_size]
            
            if patch_data.std() >= 1.0:
                patches.append({
                    'id': patch_id,
                    'image': patch_data.copy(),
                    'row': row,
                    'col': col,
                    'coords': (row, col, row+self.config.patch_size, col+self.config.patch_size)
                })
        
        logger.info(f"Created {len(patches)} patches")
        return patches, metadata
    
    def classify_patches(self, patches: List[Dict]) -> np.ndarray:
        """Run ViT classifier on patches.
        
        Args:
            patches: List of patch dictionaries
        
        Returns:
            Array of ship probabilities for each patch
        """
        if not patches:
            return np.array([])
        
        dataset = ImageDataset(patches, transform=self.vit_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        probabilities = []
        
        with torch.no_grad():
            for batch, indices in tqdm(dataloader, desc="Classifying patches"):
                batch = batch.to(self.device)
                outputs = self.vit_model(batch).squeeze()
                
                # Handle single sample case
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                probs = torch.sigmoid(outputs)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def segment_patches(self, patches: List[Dict], ship_indices: List[int]) -> Dict[int, np.ndarray]:
        """Run U-Net segmentation on ship patches.
        
        Args:
            patches: All patches
            ship_indices: Indices of patches containing ships
        
        Returns:
            Dictionary mapping patch index to segmentation mask
        """
        if not ship_indices:
            return {}
        
        # Filter ship patches
        ship_patches = [patches[i] for i in ship_indices]
        
        dataset = ImageDataset(ship_patches, transform=self.unet_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        segmentation_masks = {}
        
        with torch.no_grad():
            for batch, indices in tqdm(dataloader, desc="Segmenting patches"):
                batch = batch.to(self.device)
                outputs = self.unet_model(batch)
                masks = torch.sigmoid(outputs)
                
                for i, idx in enumerate(indices):
                    mask = masks[i, 0].cpu().numpy()  # Remove channel dimension
                    original_idx = ship_indices[idx]
                    segmentation_masks[original_idx] = mask
        
        return segmentation_masks
    
    def stitch_masks(
        self,
        patches: List[Dict],
        segmentation_masks: Dict[int, np.ndarray],
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Stitch segmentation masks back to full resolution.
        
        Args:
            patches: List of patch metadata
            segmentation_masks: Dictionary of masks
            image_shape: (height, width) of full image
        
        Returns:
            Full resolution mask
        """
        height, width = image_shape[:2]
        
        # Initialize full mask and weight map for blending
        full_mask = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Create weight kernel for smooth blending
        kernel = self._create_weight_kernel(self.config.patch_size)
        
        for patch_idx, mask in segmentation_masks.items():
            patch = patches[patch_idx]
            row, col = patch['row'], patch['col']
            
            # Apply weight kernel to mask
            weighted_mask = mask * kernel
            
            # Add to full mask with bounds checking
            end_row = min(row + self.config.patch_size, height)
            end_col = min(col + self.config.patch_size, width)
            
            mask_h = end_row - row
            mask_w = end_col - col
            
            full_mask[row:end_row, col:end_col] += weighted_mask[:mask_h, :mask_w]
            weight_map[row:end_row, col:end_col] += kernel[:mask_h, :mask_w]
        
        # Normalize by weights
        full_mask = np.divide(full_mask, weight_map, where=weight_map > 0)
        
        # Threshold to binary
        full_mask = (full_mask > 0.5).astype(np.uint8)
        
        # Post-processing
        full_mask = self._postprocess_mask(full_mask)
        
        return full_mask
    
    def _create_weight_kernel(self, size: int) -> np.ndarray:
        """Create a weight kernel for smooth blending at overlaps."""
        kernel = np.ones((size, size), dtype=np.float32)
        
        # Create linear falloff at edges
        fade_width = self.config.overlap // 2
        
        if fade_width > 0:
            for i in range(fade_width):
                weight = (i + 1) / fade_width
                kernel[i, :] *= weight
                kernel[-i-1, :] *= weight
                kernel[:, i] *= weight
                kernel[:, -i-1] *= weight
        
        return kernel
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process mask to remove small objects and smooth boundaries."""
        # Remove small connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        filtered_mask = np.zeros_like(mask)
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            if component_mask.sum() >= self.config.min_ship_size:
                filtered_mask[component_mask] = 1
        
        # Morphological operations for smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered_mask = cv2.morphologyEx(filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)
        
        return filtered_mask
    
    def extract_ship_crops(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        padding: int = 10
    ) -> List[Dict]:
        """Extract individual ship crops from image using mask.
        
        Args:
            image: Original image
            mask: Segmentation mask
            padding: Padding around detected ships
        
        Returns:
            List of ship crop dictionaries
        """
        ships = []
        
        # Find connected components (individual ships)
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        for label_id in range(1, num_labels):
            # Get ship mask
            ship_mask = (labels == label_id).astype(np.uint8)
            
            # Find bounding box
            contours, _ = cv2.findContours(ship_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # Extract crop
            ship_crop = image[y1:y2, x1:x2]
            ship_mask_crop = ship_mask[y1:y2, x1:x2]
            
            ships.append({
                'id': label_id,
                'image': ship_crop,
                'mask': ship_mask_crop,
                'bbox': (x1, y1, x2, y2),
                'area': ship_mask.sum(),
                'centroid': (x + w//2, y + h//2)
            })
        
        return ships
    
    def create_json_output(
        self,
        mask: np.ndarray,
        image_path: str,
        metadata: Dict
    ) -> Dict:
        """Create JSON output with detection results.
        
        Args:
            mask: Binary segmentation mask
            image_path: Path to original image
            metadata: Image metadata
        
        Returns:
            JSON-serializable dictionary
        """
        # Find ships
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        ships = []
        for label_id in range(1, num_labels):
            ship_mask = (labels == label_id)
            
            # Find contours
            contours, _ = cv2.findContours(
                ship_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Get polygon
            contour = contours[0].squeeze()
            if len(contour.shape) == 2 and contour.shape[0] >= 3:
                polygon = contour.tolist()
            else:
                polygon = []
            
            ships.append({
                'id': label_id,
                'bbox': [x, y, w, h],
                'area': int(ship_mask.sum()),
                'centroid': [x + w//2, y + h//2],
                'polygon': polygon
            })
        
        return {
            'image_path': image_path,
            'image_size': [metadata['width'], metadata['height']],
            'num_ships': num_labels - 1,
            'ships': ships,
            'processing_time': metadata.get('processing_time', 0),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Create visualization overlay of mask on original image.
        
        Args:
            image: Original image
            mask: Segmentation mask
            alpha: Transparency for overlay
        
        Returns:
            Image with overlay
        """
        overlay = image.copy()
        
        # Color ships in red
        mask_indices = mask > 0
        overlay[mask_indices, 0] = np.minimum(255, overlay[mask_indices, 0] + 100)
        
        # Blend with original
        result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        # Add contours in green
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # Add ship count
        num_ships = len(contours)
        cv2.putText(result, f"Ships detected: {num_ships}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result
    
    def save_outputs(
        self,
        output_dir: str,
        image_name: str,
        image: np.ndarray,
        mask: np.ndarray,
        json_data: Optional[Dict] = None,
        overlay: Optional[np.ndarray] = None,
        ship_crops: Optional[List[Dict]] = None
    ):
        """Save all outputs to disk.
        
        Args:
            output_dir: Output directory
            image_name: Base name for outputs
            image: Original image
            mask: Segmentation mask
            json_data: JSON output data
            overlay: Overlay visualization
            ship_crops: Individual ship crops
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        # Save mask
        if 'mask' in self.config.output_format:
            mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
            cv2.imwrite(mask_path, mask * 255)
            outputs['mask'] = mask_path
            logger.info(f"Saved mask to {mask_path}")
        
        # Save JSON
        if 'json' in self.config.output_format and json_data:
            json_path = os.path.join(output_dir, f"{image_name}_detections.json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            outputs['json'] = json_path
            logger.info(f"Saved JSON to {json_path}")
        
        # Save overlay
        if 'overlay' in self.config.output_format and overlay is not None:
            overlay_path = os.path.join(output_dir, f"{image_name}_overlay.jpg")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            outputs['overlay'] = overlay_path
            logger.info(f"Saved overlay to {overlay_path}")
        
        # Save ship crops
        if 'crops' in self.config.output_format and ship_crops:
            crops_dir = os.path.join(output_dir, f"{image_name}_ships")
            Path(crops_dir).mkdir(exist_ok=True)
            
            for ship in ship_crops:
                crop_path = os.path.join(crops_dir, f"ship_{ship['id']:03d}.jpg")
                cv2.imwrite(crop_path, cv2.cvtColor(ship['image'], cv2.COLOR_RGB2BGR))
                
                # Save mask too
                mask_path = os.path.join(crops_dir, f"ship_{ship['id']:03d}_mask.png")
                cv2.imwrite(mask_path, ship['mask'] * 255)
            
            outputs['crops'] = crops_dir
            logger.info(f"Saved {len(ship_crops)} ship crops to {crops_dir}")
        
        # Upload to S3 if configured
        if self.config.use_s3 and self.config.s3_bucket:
            self._upload_to_s3(outputs, image_name)
        
        return outputs
    
    def _upload_to_s3(self, outputs: Dict[str, str], image_name: str):
        """Upload outputs to S3."""
        for output_type, local_path in outputs.items():
            if os.path.isdir(local_path):
                # Upload directory contents
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        s3_key = f"ship-detection/{image_name}/{os.path.relpath(file_path, local_path)}"
                        try:
                            self.s3_client.upload_file(
                                file_path,
                                self.config.s3_bucket,
                                s3_key
                            )
                        except Exception as e:
                            logger.error(f"Failed to upload {file_path} to S3: {e}")
            else:
                # Upload single file
                s3_key = f"ship-detection/{image_name}/{os.path.basename(local_path)}"
                try:
                    self.s3_client.upload_file(
                        local_path,
                        self.config.s3_bucket,
                        s3_key
                    )
                    logger.info(f"Uploaded to s3://{self.config.s3_bucket}/{s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {output_type} to S3: {e}")
    
    def process_image(self, image_path: str, output_dir: str) -> Dict:
        """Process a single image through the full pipeline.
        
        Args:
            image_path: Path to input image (JPG/PNG)
            output_dir: Directory for outputs
        
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        image_name = Path(image_path).stem
        
        logger.info(f"Processing {image_name}...")
        
        # Step 1: Load image
        image = self.load_image(image_path)
        
        # Step 2: Tile image
        patches, metadata = self.tile_image(image)
        
        if not patches:
            logger.warning(f"No valid patches found in {image_name}")
            return {
                'image': image_name,
                'status': 'no_valid_patches',
                'processing_time': time.time() - start_time
            }
        
        # Step 3: Classify patches
        probabilities = self.classify_patches(patches)
        
        # Step 4: Filter ship patches
        ship_indices = np.where(probabilities >= self.config.confidence_threshold)[0]
        logger.info(f"Found {len(ship_indices)}/{len(patches)} patches with ships")
        
        # Save patch predictions if requested
        if self.config.save_patch_predictions:
            patch_results = pd.DataFrame({
                'patch_id': [p['id'] for p in patches],
                'row': [p['row'] for p in patches],
                'col': [p['col'] for p in patches],
                'probability': probabilities,
                'has_ship': probabilities >= self.config.confidence_threshold
            })
            patch_results.to_csv(
                os.path.join(output_dir, f"{image_name}_patches.csv"),
                index=False
            )
        
        # Step 5: Segment ship patches
        if len(ship_indices) > 0:
            segmentation_masks = self.segment_patches(patches, ship_indices.tolist())
        else:
            segmentation_masks = {}
        
        # Step 6: Stitch masks
        full_mask = self.stitch_masks(
            patches,
            segmentation_masks,
            (metadata['height'], metadata['width'])
        )
        
        # Step 7: Extract ship crops
        ship_crops = None
        if 'crops' in self.config.output_format:
            ship_crops = self.extract_ship_crops(image, full_mask)
        
        # Step 8: Create outputs
        metadata['processing_time'] = time.time() - start_time
        
        json_data = None
        if 'json' in self.config.output_format:
            json_data = self.create_json_output(full_mask, image_path, metadata)
        
        overlay = None
        if 'overlay' in self.config.output_format:
            overlay = self.create_overlay(image, full_mask)
        
        # Step 9: Save outputs
        outputs = self.save_outputs(
            output_dir,
            image_name,
            image,
            full_mask,
            json_data,
            overlay,
            ship_crops
        )
        
        # Processing stats
        num_ships = len(ship_crops) if ship_crops else cv2.connectedComponents(full_mask)[0] - 1
        
        stats = {
            'image': image_name,
            'status': 'success',
            'processing_time': metadata['processing_time'],
            'image_size': (metadata['width'], metadata['height']),
            'num_patches': len(patches),
            'num_ship_patches': len(ship_indices),
            'num_ships_detected': num_ships,
            'outputs': outputs
        }
        
        logger.info(f"Completed {image_name} in {metadata['processing_time']:.2f} seconds")
        logger.info(f"Detected {num_ships} ships")
        
        return stats
    
    def process_batch(self, image_paths: List[str], output_dir: str) -> List[Dict]:
        """Process multiple images.
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory
        
        Returns:
            List of processing results
        """
        results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.process_image(image_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'image': Path(image_path).stem,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save batch summary
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch processing complete. Summary saved to {summary_path}")
        
        return results


@click.command()
@click.option('--image', '-i', help='Path to input image (JPG/PNG)')
@click.option('--batch', '-b', help='Path to directory with images for batch processing')
@click.option('--output-dir', '-o', default='./outputs', help='Output directory')
@click.option('--vit-checkpoint', required=True, help='Path to ViT model checkpoint')
@click.option('--vit-config', required=True, help='Path to ViT config YAML')
@click.option('--unet-checkpoint', required=True, help='Path to U-Net model checkpoint')
@click.option('--unet-config', required=True, help='Path to U-Net config YAML')
@click.option('--patch-size', default=224, help='Patch size for tiling')
@click.option('--overlap', default=32, help='Overlap between patches')
@click.option('--batch-size', default=32, help='Batch size for inference')
@click.option('--confidence', default=0.5, help='Confidence threshold for ship detection')
@click.option('--device', default='cuda', help='Device for inference (cuda/cpu)')
@click.option('--formats', '-f', multiple=True, default=['mask', 'json', 'overlay'], 
              help='Output formats (mask/json/overlay/crops)')
@click.option('--save-patches', is_flag=True, help='Save patch-level predictions')
@click.option('--use-s3', is_flag=True, help='Upload results to S3')
@click.option('--s3-bucket', help='S3 bucket name')
def main(image, batch, output_dir, vit_checkpoint, vit_config, unet_checkpoint, 
         unet_config, patch_size, overlap, batch_size, confidence, device,
         formats, save_patches, use_s3, s3_bucket):
    """Ship detection and segmentation inference pipeline for JPG/PNG images."""
    
    if not image and not batch:
        click.echo("Error: Please provide either --image or --batch")
        return
    
    # Create configuration
    config = PipelineConfig(
        vit_checkpoint=vit_checkpoint,
        vit_config=vit_config,
        unet_checkpoint=unet_checkpoint,
        unet_config=unet_config,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        confidence_threshold=confidence,
        device=device,
        use_s3=use_s3,
        s3_bucket=s3_bucket,
        output_format=list(formats),
        save_patch_predictions=save_patches
    )
    
    # Create pipeline
    pipeline = InferencePipeline(config)
    
    # Process image(s)
    if image:
        # Single image processing
        stats = pipeline.process_image(image, output_dir)
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Image: {stats['image']}")
        print(f"Status: {stats['status']}")
        print(f"Time: {stats.get('processing_time', 0):.2f} seconds")
        print(f"Image size: {stats.get('image_size', 'N/A')}")
        print(f"Patches processed: {stats.get('num_patches', 0)}")
        print(f"Ship patches: {stats.get('num_ship_patches', 0)}")
        print(f"Ships detected: {stats.get('num_ships_detected', 0)}")
        print(f"Outputs saved to: {output_dir}")
        print("="*60)
        
    else:
        # Batch processing
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(batch).glob(f"*{ext}"))
            image_paths.extend(Path(batch).glob(f"*{ext.upper()}"))
        
        if not image_paths:
            click.echo(f"No images found in {batch}")
            return
        
        click.echo(f"Found {len(image_paths)} images to process")
        results = pipeline.process_batch([str(p) for p in image_paths], output_dir)
        
        # Print summary
        successful = sum(1 for r in results if r.get('status') == 'success')
        total_ships = sum(r.get('num_ships_detected', 0) for r in results)
        total_time = sum(r.get('processing_time', 0) for r in results)
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Images processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Total ships detected: {total_ships}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/len(results):.2f} seconds")
        print(f"Outputs saved to: {output_dir}")
        print("="*60)


if __name__ == "__main__":
    main()