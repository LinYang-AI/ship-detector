import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """Convert RLE string to binary mask
    
    Args:
        mask_rle: Run-length encoded string
        shape: (height, width) of target image
    
    Returs:
        Binary mask array
    """
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    s = mask_rle.split()  # Split RLE into components
    starts = np.array(s[0::2], dtype=int) - 1 # RLE uses 1-based indexing
    lengths = np.array(s[1::2], dtype=int)
    ends = starts + lengths
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    
    return img.reshape(shape).T  # RLE uses column-major order


def tile_geotiff(
    tiff_path: str,
    output_dir: str,
    patch_size: int = 224,
    overlap: int = 32
) -> List[Dict]:
    """Tile a GeoTIFF into patches preserving georeferencing.
    
    Args:
        tiff_path: Path to input GeoTIFF
        output_dir: Directory for output patches
        patch_size: Size of square patches
        overlap: Overlap between patches in pixels
    
    Returs:
        List of patch metadata dictionaries
    """
    patches = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(tiff_path) as src:
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

        stride = patch_size - overlap
        patch_id = 0
        
        for row in range(0, height - patch_size + 1, stride):
            for col in range(0, width - patch_size + 1, stride):
                # Define window and read patch
                window = Window(col, row, patch_size, patch_size)
                patch = src.read(window=window)
                
                # Skip empty patches (all zeros or very low variance)
                if patch.std() < 0.01:
                    continue
                
                # Get georeferencing for this patch
                patch_transform = rasterio.windows.transform(window, transform)
                
                # Convert to bounds (minx, miny, maxx, maxy)
                bounds = rasterio.windows.bounds(window, transform)
                
                # Save patch
                patch_name = f"{Path(tiff_path).stem}_patch_{patch_id:06d}.tif"
                patch_path = os.path.join(output_dir, patch_name)
                
                # Write patch with georeferencing
                with rasterio.open(
                    patch_path, 'w',
                    driver='GTiff',
                    height=patch_size,
                    width=patch_size,
                    count=src.count,
                    dtype=patch.dtype,
                    crs=crs,
                    transform=patch_transform,
                    compress='lzw'
                ) as dst:
                    dst.write(patch)
                
                # Store metadata
                patches.append({
                    'patch_id':patch_id,
                    'patch_path': patch_path,
                    'source_image': Path(tiff_path).stem,
                    'row': row,
                    'col': col,
                    'minx': bounds[0],
                    'miny': bounds[1],
                    'maxx': bounds[2],
                    'maxy': bounds[3],
                    'crs': str(crs),
                    'width': patch_size,
                    'height': patch_size
                })
                
                patch_id += 1
    
    print(f"Created {len(patches)} patches from {tiff_path}")
    return patches

def process_masks(
    mask_df: pd.DataFrame,
    image_shape: Tuple[int, int],
    patches_metadata: List[Dict],
    output_dir: str
) -> pd.DataFrame:
    """Process RLE masks and assign to patches.
    
    Args:
        mask_df: DataFrame with ImageId and EncodedPixels columns
        image_shape: (height, width) of full image
        patches_metadata: List of patch metadata
        output_dir: Directory for mask outputs
    
    Returns:
        Updated patches DataFrame with ship labels
    """
    # Decode full image mask
    full_mask = np.zeros(image_shape, dtype=np.uint8)
    
    for _, row in mask_df.iterrows():
        if pd.notna(row['EncodedPixels']):
            ship_mask = rle_decode(row['EncodedPixels'], image_shape)
            full_mask = np.maximum(full_mask, ship_mask)
        
    # Check each patch for ships
    for patch in patches_metadata:
        row, col = patch['row'], patch['col']
        patch_size = patch['width']
        
        # Extract patch from full mask
        patch_mask = full_mask[row:row+patch_size, col: col+patch_size]
        
        # Label patch (has_ship if >1% pixels are ships)
        ship_pixels = np.sum(patch_mask)
        total_pixels = patch_size * patch_size
        patch['has_ship'] = int(ship_pixels > total_pixels * 0.01)
        patch['ship_pixel_ratio'] = ship_pixels / total_pixels
        
        # Save mask if has ships
        if patch['has_ship']:
            mask_path = patch['patch_path'].replace('.tif', '_mask.png')
            cv2.imwrite(mask_path, patch_mask * 255)
            patch['mask_path'] = mask_path
        else:
            patch['mask_path'] = None

    return pd.DataFrame(patches_metadata)


def create_synthetic_test():
    """Create synthetic test data for validation"""
    print("Creating synthetic data...")
    
    # Create synthetic 512x512 image
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Random noise image with some structure
    np.random.seed(42)
    img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    
    # Add some "ships" (bright rectangles)
    img[100:150, 100:180] = 255  # Ship 1
    img[300:340, 250:320] = 255  # Ship 2

    # Save synthetic image
    cv2.imwrite(str(test_dir / 'synthetic.tif'), img)
    
    # Create RLE for the ships
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:150, 100:180] = 1
    mask[300:340, 250:320] = 1
    
    # Convert to RLE (column-major order)
    mask_t = mask.T.flatten()
    runs = []
    start = None
    for i, val in enumerate(mask_t):
        if val == 1 and start is None:
            start = i + 1
        elif val == 0 and start is not None:
            length = i - start + 1
            runs.extend([start, length])
            start = None
    
    if start is not None:  # Handle case where mask ends with 1
        runs.extend([start, len(mask_t) - start + 1])
        
    rle_string = ' '.join(map(str, runs))
    
    # Save metadata
    test_csv = pd.DataFrame({
        'ImageId': ['synthetic.tif'],
        'EncodedPixels': [rle_string]
    })
    test_csv.to_csv(test_dir / 'synthetic_masks.csv', index=False)
    
    print(f"Created synthetic test data in {test_dir}")
    return str(test_dir)
