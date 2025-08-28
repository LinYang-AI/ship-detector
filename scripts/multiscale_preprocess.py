import cv2
import numpy as np
from typing import Tuple, List
from PIL import Image
import torch
from torchvision import transforms

def multiscale_resize_with_context(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Multi-scale approach: combine different resolution views
    
    Args:
        image: Input image (768x768x3)
        target_size: Target size (224)
    
    Returns:
        Processed image (224x224x3)
    """
    h, w = image.shape[:2]
    
    # Scale 1: Full image (global context)
    # Direct resize to 112x112, preserves large ships
    global_view = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    
    # Scale 2: Center crop (medium detail)
    # Take center 384x384 and resize to 112x112
    center = h // 2
    crop_size = 384
    start = center - crop_size // 2
    center_crop = image[start:start+crop_size, start:start+crop_size]
    medium_view = cv2.resize(center_crop, (112, 112), interpolation=cv2.INTER_AREA)
    
    # Combine into 224x224
    # Top half: global view, bottom half: medium view
    combined = np.zeros((224, 224, 3), dtype=image.dtype)
    combined[:112, :112] = global_view
    combined[:112, 112:] = medium_view
    combined[112:, :112] = medium_view  # Repeat for symmetry
    combined[112:, 112:] = global_view
    
    return combined

def adaptive_resize_preserve_ships(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Adaptive resize focusing on ship preservation
    
    Args:
        image: Input image (768x768x3)
        target_size: Target size (224)
    
    Returns:
        Processed image (224x224x3)
    """
    # Method: Use INTER_AREA for downsampling (better for small objects)
    # Add slight sharpening to enhance small ship edges
    
    # Step 1: Resize with INTER_AREA (best for downsampling)
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Step 2: Mild sharpening to enhance small objects
    kernel = np.array([[-0.1, -0.1, -0.1],
                       [-0.1,  1.8, -0.1],
                       [-0.1, -0.1, -0.1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    # Step 3: Blend original and sharpened (subtle enhancement)
    enhanced = cv2.addWeighted(resized, 0.7, sharpened, 0.3, 0)
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def sliding_window_ensemble(image: np.ndarray, window_size: int = 384, stride: int = 192) -> List[np.ndarray]:
    """
    Generate multiple 224x224 patches from sliding windows
    For ensemble prediction during inference
    
    Args:
        image: Input image (768x768x3)
        window_size: Sliding window size
        stride: Sliding stride
    
    Returns:
        List of 224x224 patches
    """
    h, w = image.shape[:2]
    patches = []
    
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            # Extract window
            window = image[y:y+window_size, x:x+window_size]
            
            # Resize to 224x224
            patch = cv2.resize(window, (224, 224), interpolation=cv2.INTER_AREA)
            patches.append(patch)
    
    return patches

class MultiScaleShipDataset(torch.utils.data.Dataset):
    """Dataset with multi-scale preprocessing"""
    
    def __init__(self, manifest_df, transform=None, preprocessing_method='multiscale'):
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.preprocessing_method = preprocessing_method
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # Load 768x768 image
        image = cv2.imread(row['patch_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        if self.preprocessing_method == 'multiscale':
            image = multiscale_resize_with_context(image)
        elif self.preprocessing_method == 'adaptive':
            image = adaptive_resize_preserve_ships(image)
        else:
            # Standard resize
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['has_ship'], dtype=torch.float32)
        return image, label

# Test function
def test_preprocessing_methods():
    """Test different preprocessing methods on synthetic data"""
    
    # Create synthetic 768x768 image with small ship
    image = np.random.randint(50, 150, (768, 768, 3), dtype=np.uint8)
    
    # Add small ship (8x8 pixels)
    image[100:108, 200:208] = 255
    
    # Add medium ship (20x20 pixels)  
    image[400:420, 500:520] = 255
    
    # Test methods
    methods = {
        'Standard Resize': lambda img: cv2.resize(img, (224, 224)),
        'Adaptive Resize': adaptive_resize_preserve_ships,
        'Multi-scale': multiscale_resize_with_context
    }
    
    results = {}
    for name, method in methods.items():
        processed = method(image.copy())
        
        # Simple ship detection: count high-intensity pixels
        ship_pixels = np.sum(processed > 200)
        results[name] = {
            'ship_pixels': ship_pixels,
            'shape': processed.shape,
            'max_intensity': processed.max()
        }
    
    return results

if __name__ == "__main__":
    # Test preprocessing methods
    results = test_preprocessing_methods()
    
    print("Preprocessing Method Comparison:")
    print("-" * 40)
    for method, metrics in results.items():
        print(f"{method}:")
        print(f"  Ship pixels: {metrics['ship_pixels']}")
        print(f"  Max intensity: {metrics['max_intensity']}")
        print(f"  Shape: {metrics['shape']}")
        print()