from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchvision import transforms
from scripts.train_unet import UNetShipSegmentation
from scripts.train_vit import ViTShipClassifier
import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import pandas as pd
from tqdm import tqdm
from shapely.geometry import shape, mapping, box, Polygon
from shapely.ops import unary_union
import geojson

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    output_format: List[str] = None  # ['mask', 'geojson', 'overlay']

    def __post_init__(self):
        if self.output_format is None:
            self.output_format = ['mask', 'geojson']


class TiledDataset(Dataset):
    """Dataset for tiled image patches."""

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
                from PIL import Image
                image = Image.fromarray(image.astype(np.uint8))
                image = self.transform(image)

        return image, idx


class InferencePipeline:
    """Main inference pipeline combining ViT and U-Net."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu')

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

    def _load_vit_model(self):
        """Load ViT classifier model."""
        import yaml
        with open(self.config.vit_config, 'r') as f:
            vit_config = yaml.safe_load(f)

        model = ViTShipClassifier(vit_config)

        if self.config.vit_checkpoint.endswith('.ckpt'):
            checkpoint = torch.load(
                self.config.vit_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(torch.load(
                self.config.vit_checkpoint, map_location='cpu'))

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
            checkpoint = torch.load(
                self.config.unet_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(torch.load(
                self.config.unet_checkpoint, map_location='cpu'))

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

    def tile_image(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """Tile a GeoTIFF into patches.

        Returns:
            Tuple of (patches_list, metadata)
        """
        patches = []

        with rasterio.open(image_path) as src:
            # Store metadata
            metadata = {
                'width': src.width,
                'height': src.height,
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds,
                'count': src.count
            }

            # Calculate tiling
            stride = self.config.patch_size - self.config.overlap
            patch_id = 0

            logger.info(f"Tiling image: {src.width}x{src.height}")

            for row in range(0, src.height - self.config.patch_size + 1, stride):
                for col in range(0, src.width - self.config.patch_size + 1, stride):
                    # Define window
                    window = Window(
                        col, row, self.config.patch_size, self.config.patch_size)

                    # Read patch
                    patch_data = src.read(window=window)

                    # Convert to RGB if needed
                    if patch_data.shape[0] == 1:
                        patch_data = np.repeat(patch_data, 3, axis=0)
                    elif patch_data.shape[0] > 3:
                        patch_data = patch_data[:3]

                    # Transpose to HWC
                    patch_data = np.transpose(patch_data, (1, 2, 0))

                    # Skip empty patches
                    if patch_data.std() < 0.01:
                        continue

                    # Get georeferencing
                    patch_transform = rasterio.windows.transform(
                        window, src.transform)
                    bounds = rasterio.windows.bounds(window, src.transform)

                    patches.append({
                        'id': patch_id,
                        'image': patch_data,
                        'row': row,
                        'col': col,
                        'window': window,
                        'transform': patch_transform,
                        'bounds': bounds
                    })

                    patch_id += 1

            # Handle edge cases (partial patches at borders)
            # Add right edge patches
            if src.width % stride != 0:
                col = src.width - self.config.patch_size
                for row in range(0, src.height - self.config.patch_size + 1, stride):
                    window = Window(
                        col, row, self.config.patch_size, self.config.patch_size)
                    patch_data = src.read(window=window)

                    if patch_data.shape[0] == 1:
                        patch_data = np.repeat(patch_data, 3, axis=0)
                    elif patch_data.shape[0] > 3:
                        patch_data = patch_data[:3]

                    patch_data = np.transpose(patch_data, (1, 2, 0))

                    if patch_data.std() >= 0.01:
                        patches.append({
                            'id': patch_id,
                            'image': patch_data,
                            'row': row,
                            'col': col,
                            'window': window,
                            'transform': rasterio.windows.transform(window, src.transform),
                            'bounds': rasterio.windows.bounds(window, src.transform)
                        })
                        patch_id += 1

            # Add bottom edge patches
            if src.height % stride != 0:
                row = src.height - self.config.patch_size
                for col in range(0, src.width - self.config.patch_size + 1, stride):
                    window = Window(
                        col, row, self.config.patch_size, self.config.patch_size)
                    patch_data = src.read(window=window)

                    if patch_data.shape[0] == 1:
                        patch_data = np.repeat(patch_data, 3, axis=0)
                    elif patch_data.shape[0] > 3:
                        patch_data = patch_data[:3]

                    patch_data = np.transpose(patch_data, (1, 2, 0))

                    if patch_data.std() >= 0.01:
                        patches.append({
                            'id': patch_id,
                            'image': patch_data,
                            'row': row,
                            'col': col,
                            'window': window,
                            'transform': rasterio.windows.transform(window, src.transform),
                            'bounds': rasterio.windows.bounds(window, src.transform)
                        })
                        patch_id += 1

        logger.info(f"Created {len(patches)} patches")
        return patches, metadata

    def classify_patches(self, patches: List[Dict]) -> np.ndarray:
        """Run ViT classifier on patches.

        Returns:
            Array of ship probabilities for each patch
        """
        dataset = TiledDataset(patches, transform=self.vit_transform)
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
                probs = torch.sigmoid(outputs)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def segment_patches(self, patches: List[Dict], ship_indices: List[int]) -> Dict[int, np.ndarray]:
        """Run U-Net segmentation on ship patches.

        Returns:
            Dictionary mapping patch index to segmentation mask
        """
        # Filter patches
        ship_patches = [patches[i] for i in ship_indices]

        if not ship_patches:
            return {}

        dataset = TiledDataset(ship_patches, transform=self.unet_transform)
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
                    # Remove channel dimension
                    mask = masks[i, 0].cpu().numpy()
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
        height, width = image_shape

        # Initialize full mask and weight map for blending
        full_mask = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)

        # Create weight kernel for smooth blending
        kernel_size = self.config.patch_size
        kernel = self._create_weight_kernel(kernel_size)

        for patch_idx, mask in segmentation_masks.items():
            patch = patches[patch_idx]
            row, col = patch['row'], patch['col']

            # Apply weight kernel to mask
            weighted_mask = mask * kernel

            # Add to full mask
            end_row = min(row + kernel_size, height)
            end_col = min(col + kernel_size, width)

            # Handle edge cases where patch extends beyond image
            mask_h = end_row - row
            mask_w = end_col - col

            full_mask[row:end_row,
                      col:end_col] += weighted_mask[:mask_h, :mask_w]
            weight_map[row:end_row, col:end_col] += kernel[:mask_h, :mask_w]

        # Normalize by weights
        full_mask = np.divide(full_mask, weight_map, where=weight_map > 0)

        # Threshold to binary
        full_mask = (full_mask > 0.5).astype(np.uint8)

        # Post-processing: remove small objects
        full_mask = self._postprocess_mask(full_mask)

        return full_mask

    def _create_weight_kernel(self, size: int) -> np.ndarray:
        """Create a weight kernel for smooth blending at overlaps."""
        kernel = np.ones((size, size), dtype=np.float32)

        # Create linear falloff at edges
        fade_width = self.config.overlap // 2

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
        filtered_mask = cv2.morphologyEx(
            filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)

        return filtered_mask

    def mask_to_geojson(
        self,
        mask: np.ndarray,
        transform: Any,
        crs: Any,
        simplify_tolerance: float = 1.0
    ) -> Dict:
        """Convert binary mask to GeoJSON with georeferencing.

        Args:
            mask: Binary mask
            transform: Rasterio transform
            crs: Coordinate reference system
            simplify_tolerance: Polygon simplification tolerance

        Returns:
            GeoJSON dictionary
        """
        from rasterio import features
        from shapely.geometry import shape, mapping
        from shapely.ops import transform as shapely_transform

        # Extract shapes from mask
        shapes_generator = features.shapes(
            mask.astype(np.uint8),
            transform=transform
        )

        features_list = []
        for geom, value in shapes_generator:
            if value == 1:  # Only ship pixels
                # Convert to Shapely geometry
                poly = shape(geom)

                # Simplify if requested
                if simplify_tolerance > 0:
                    poly = poly.simplify(
                        simplify_tolerance, preserve_topology=True)

                # Calculate properties
                area = poly.area
                perimeter = poly.length

                # Get bounding box
                minx, miny, maxx, maxy = poly.bounds

                # Create feature
                feature = {
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "class": "ship",
                        "area_m2": area,
                        "perimeter_m": perimeter,
                        "bbox": [minx, miny, maxx, maxy],
                        "confidence": 1.0  # Could add actual confidence if stored
                    }
                }
                features_list.append(feature)

        # Create FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": features_list,
            "crs": {
                "type": "name",
                "properties": {
                    "name": str(crs) if crs else "EPSG:4326"
                }
            }
        }

        return geojson

    def create_overlay(
        self,
        image_path: str,
        mask: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Create visualization overlay of mask on original image."""
        # Read original image
        with rasterio.open(image_path) as src:
            image = src.read()

            # Convert to RGB
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] > 3:
                image = image[:3]

            # Transpose to HWC
            image = np.transpose(image, (1, 2, 0))

            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() -
                     image.min()) * 255).astype(np.uint8)

        # Create colored mask
        overlay = image.copy()
        mask_indices = mask > 0
        overlay[mask_indices, 0] = np.minimum(
            255, overlay[mask_indices, 0] + 100)  # Add red

        # Blend
        result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)

        # Add contours
        contours, _ = cv2.findContours(mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result

    def save_outputs(
        self,
        output_dir: str,
        image_name: str,
        mask: np.ndarray,
        geojson_data: Optional[Dict] = None,
        overlay: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        """Save all outputs to disk or S3."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Save mask
        if 'mask' in self.config.output_format:
            mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
            cv2.imwrite(mask_path, mask * 255)
            outputs['mask'] = mask_path
            logger.info(f"Saved mask to {mask_path}")

        # Save GeoJSON
        if 'geojson' in self.config.output_format and geojson_data:
            geojson_path = os.path.join(
                output_dir, f"{image_name}_ships.geojson")
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            outputs['geojson'] = geojson_path
            logger.info(f"Saved GeoJSON to {geojson_path}")

        # Save overlay
        if 'overlay' in self.config.output_format and overlay is not None:
            overlay_path = os.path.join(
                output_dir, f"{image_name}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            outputs['overlay'] = overlay_path
            logger.info(f"Saved overlay to {overlay_path}")

        # Save metadata
        if metadata:
            metadata_path = os.path.join(
                output_dir, f"{image_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                # Convert non-serializable objects
                clean_metadata = {
                    k: str(v) if not isinstance(
                        v, (str, int, float, list, dict)) else v
                    for k, v in metadata.items()
                }
                json.dump(clean_metadata, f, indent=2)
            outputs['metadata'] = metadata_path

        # Upload to S3 if configured
        if self.config.use_s3 and self.config.s3_bucket:
            self._upload_to_s3(outputs, image_name)

        return outputs

    def _upload_to_s3(self, outputs: Dict[str, str], image_name: str):
        """Upload outputs to S3."""
        for output_type, local_path in outputs.items():
            s3_key = f"ship-detection/{image_name}/{os.path.basename(local_path)}"

            try:
                self.s3_client.upload_file(
                    local_path,
                    self.config.s3_bucket,
                    s3_key
                )
                logger.info(
                    f"Uploaded {output_type} to s3://{self.config.s3_bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload {output_type} to S3: {e}")

    def process_image(self, image_path: str, output_dir: str) -> Dict:
        """Process a single image through the full pipeline.

        Args:
            image_path: Path to input GeoTIFF
            output_dir: Directory for outputs

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        image_name = Path(image_path).stem

        logger.info(f"Processing {image_name}...")

        # Step 1: Tile image
        patches, metadata = self.tile_image(image_path)

        # Step 2: Classify patches
        probabilities = self.classify_patches(patches)

        # Step 3: Filter ship patches
        ship_indices = np.where(
            probabilities >= self.config.confidence_threshold)[0]
        logger.info(
            f"Found {len(ship_indices)}/{len(patches)} patches with ships")

        # Step 4: Segment ship patches
        if len(ship_indices) > 0:
            segmentation_masks = self.segment_patches(
                patches, ship_indices.tolist())
        else:
            segmentation_masks = {}

        # Step 5: Stitch masks
        full_mask = self.stitch_masks(
            patches,
            segmentation_masks,
            (metadata['height'], metadata['width'])
        )

        # Step 6: Convert to GeoJSON
        geojson_data = None
        if 'geojson' in self.config.output_format:
            geojson_data = self.mask_to_geojson(
                full_mask,
                metadata['transform'],
                metadata['crs']
            )
            num_ships = len(geojson_data['features'])
            logger.info(f"Detected {num_ships} ships")

        # Step 7: Create overlay
        overlay = None
        if 'overlay' in self.config.output_format:
            overlay = self.create_overlay(image_path, full_mask)

        # Step 8: Save outputs
        outputs = self.save_outputs(
            output_dir,
            image_name,
            full_mask,
            geojson_data,
            overlay,
            metadata
        )

        # Processing stats
        processing_time = time.time() - start_time
        stats = {
            'image': image_name,
            'processing_time': processing_time,
            'num_patches': len(patches),
            'num_ship_patches': len(ship_indices),
            'num_ships_detected': len(geojson_data['features']) if geojson_data else 0,
            'outputs': outputs
        }

        logger.info(f"Completed {image_name} in {processing_time:.2f} seconds")

        return stats


@click.command()
@click.option('--image', '-i', required=True, help='Path to input GeoTIFF image')
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
@click.option('--formats', '-f', multiple=True, default=['mask', 'geojson'],
              help='Output formats (mask/geojson/overlay)')
@click.option('--use-s3', is_flag=True, help='Upload results to S3')
@click.option('--s3-bucket', help='S3 bucket name')
def main(image, output_dir, vit_checkpoint, vit_config, unet_checkpoint,
         unet_config, patch_size, overlap, batch_size, confidence, device,
         formats, use_s3, s3_bucket):
    """Ship detection and segmentation inference pipeline."""

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
        output_format=list(formats)
    )

    # Create pipeline
    pipeline = InferencePipeline(config)

    # Process image
    stats = pipeline.process_image(image, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Image: {stats['image']}")
    print(f"Time: {stats['processing_time']:.2f} seconds")
    print(f"Patches processed: {stats['num_patches']}")
    print(f"Ship patches: {stats['num_ship_patches']}")
    print(f"Ships detected: {stats['num_ships_detected']}")
    print(f"Outputs saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
