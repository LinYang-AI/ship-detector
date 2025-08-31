import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class PatchStitcher:
    """Advanced patch stitching with multiple blending strategies."""
    
    def __init__(
        self,
        image_shape: Tuple[int, int],
        patch_size: int = 224,
        overlap: int = 32,
        blend_mode: str = 'weighted'  # 'weighted', 'max', 'average', 'feather'
    ):
        """
        Args:
            image_shape: (height, width) of full image
            patch_size: Size of each patch
            overlap: Overlap between patches
            blend_mode: Blending strategy for overlapping regions
        """
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.overlap = overlap
        self.blend_mode = blend_mode
        
        # Initialize accumulation arrays
        self.reset()
    
    def reset(self):
        """Reset accumulation arrays for new stitching."""
        h, w = self.image_shape
        self.accumulated_mask = np.zeros((h, w), dtype=np.float32)
        self.weight_map = np.zeros((h, w), dtype=np.float32)
        self.confidence_map = np.zeros((h, w), dtype=np.float32)
        self.patch_count = np.zeros((h, w), dtype=np.int32)
    
    def add_patch(
        self,
        mask: np.ndarray,
        position: Tuple[int, int],
        confidence: float = 1.0
    ):
        """Add a patch to the accumulation.
        
        Args:
            mask: Patch mask to add
            position: (row, col) position in full image
            confidence: Confidence score for this patch
        """
        row, col = position
        h, w = self.image_shape
        patch_h, patch_w = mask.shape[:2]
        
        # Calculate valid region (handle edge cases)
        end_row = min(row + patch_h, h)
        end_col = min(col + patch_w, w)
        valid_h = end_row - row
        valid_w = end_col - col
        
        # Get weight kernel for this patch
        weight_kernel = self._get_weight_kernel(mask.shape, confidence)
        
        # Add to accumulation based on blend mode
        if self.blend_mode == 'weighted':
            self.accumulated_mask[row:end_row, col:end_col] += \
                mask[:valid_h, :valid_w] * weight_kernel[:valid_h, :valid_w]
            self.weight_map[row:end_row, col:end_col] += \
                weight_kernel[:valid_h, :valid_w]
        
        elif self.blend_mode == 'max':
            self.accumulated_mask[row:end_row, col:end_col] = np.maximum(
                self.accumulated_mask[row:end_row, col:end_col],
                mask[:valid_h, :valid_w]
            )
        
        elif self.blend_mode == 'average':
            self.accumulated_mask[row:end_row, col:end_col] += mask[:valid_h, :valid_w]
            self.patch_count[row:end_row, col:end_col] += 1
        
        elif self.blend_mode == 'feather':
            feathered = self._feather_blend(
                self.accumulated_mask[row:end_row, col:end_col],
                mask[:valid_h, :valid_w],
                weight_kernel[:valid_h, :valid_w]
            )
            self.accumulated_mask[row:end_row, col:end_col] = feathered
        
        # Update confidence map
        self.confidence_map[row:end_row, col:end_col] = np.maximum(
            self.confidence_map[row:end_row, col:end_col],
            confidence
        )
    
    def _get_weight_kernel(
        self,
        shape: Tuple[int, int],
        confidence: float = 1.0
    ) -> np.ndarray:
        """Create weight kernel for smooth blending.
        
        Args:
            shape: Shape of the patch
            confidence: Confidence multiplier
        
        Returns:
            Weight kernel
        """
        h, w = shape[:2]
        kernel = np.ones((h, w), dtype=np.float32) * confidence
        
        # Distance transform from edges
        fade_width = self.overlap // 2
        
        if fade_width > 0:
            # Create distance map from edges
            edge_mask = np.ones((h, w), dtype=np.uint8)
            edge_mask[fade_width:-fade_width, fade_width:-fade_width] = 0
            
            # Distance transform
            dist = cv2.distanceTransform(
                1 - edge_mask,
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE
            )
            
            # Normalize to 0-1
            dist = np.clip(dist / fade_width, 0, 1)
            kernel *= dist
        
        return kernel
    
    def _feather_blend(
        self,
        existing: np.ndarray,
        new_patch: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """Feather blending for smooth transitions.
        
        Args:
            existing: Existing accumulated mask region
            new_patch: New patch to blend
            weights: Weight kernel
        
        Returns:
            Blended result
        """
        # Gaussian blur the weights for smoother transition
        smooth_weights = gaussian_filter(weights, sigma=2.0)
        
        # Blend based on smoothed weights
        result = existing * (1 - smooth_weights) + new_patch * smooth_weights
        
        return result
    
    def get_final_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Get the final stitched mask.
        
        Args:
            threshold: Threshold for binarization
        
        Returns:
            Final binary mask
        """
        if self.blend_mode == 'weighted':
            # Normalize by weights
            final = np.divide(
                self.accumulated_mask,
                self.weight_map,
                out=np.zeros_like(self.accumulated_mask),
                where=self.weight_map > 0
            )
        
        elif self.blend_mode == 'average':
            # Normalize by patch count
            final = np.divide(
                self.accumulated_mask,
                self.patch_count,
                out=np.zeros_like(self.accumulated_mask),
                where=self.patch_count > 0
            )
        
        else:
            final = self.accumulated_mask
        
        # Apply threshold
        binary_mask = (final > threshold).astype(np.uint8)
        
        return binary_mask
    
    def resolve_conflicts(
        self,
        patches: List[Dict],
        method: str = 'confidence'
    ) -> np.ndarray:
        """Resolve conflicts in overlapping regions.
        
        Args:
            patches: List of patch dictionaries with masks and metadata
            method: Conflict resolution method ('confidence', 'voting', 'consensus')
        
        Returns:
            Resolved mask
        """
        if method == 'confidence':
            # Use highest confidence prediction
            for patch in sorted(patches, key=lambda x: x.get('confidence', 0)):
                self.add_patch(
                    patch['mask'],
                    (patch['row'], patch['col']),
                    patch.get('confidence', 1.0)
                )
        
        elif method == 'voting':
            # Majority voting in overlapping regions
            vote_map = np.zeros(self.image_shape, dtype=np.float32)
            vote_count = np.zeros(self.image_shape, dtype=np.int32)
            
            for patch in patches:
                row, col = patch['row'], patch['col']
                mask = patch['mask']
                h, w = mask.shape[:2]
                
                end_row = min(row + h, self.image_shape[0])
                end_col = min(col + w, self.image_shape[1])
                
                vote_map[row:end_row, col:end_col] += mask[:end_row-row, :end_col-col]
                vote_count[row:end_row, col:end_col] += 1
            
            # Threshold based on majority
            final_mask = (vote_map > vote_count / 2).astype(np.uint8)
            return final_mask
        
        elif method == 'consensus':
            # Require agreement from multiple patches
            consensus_threshold = 0.7  # 70% agreement required
            
            agreement_map = np.zeros(self.image_shape, dtype=np.float32)
            coverage_map = np.zeros(self.image_shape, dtype=np.int32)
            
            for patch in patches:
                row, col = patch['row'], patch['col']
                mask = patch['mask']
                h, w = mask.shape[:2]
                
                end_row = min(row + h, self.image_shape[0])
                end_col = min(col + w, self.image_shape[1])
                
                agreement_map[row:end_row, col:end_col] += mask[:end_row-row, :end_col-col]
                coverage_map[row:end_row, col:end_col] += 1
            
            # Calculate consensus
            consensus = np.divide(
                agreement_map,
                coverage_map,
                out=np.zeros_like(agreement_map),
                where=coverage_map > 0
            )
            
            final_mask = (consensus >= consensus_threshold).astype(np.uint8)
            return final_mask
        
        return self.get_final_mask()


class BoundaryRefinement:
    """Refine boundaries between stitched patches."""
    
    @staticmethod
    def smooth_boundaries(
        mask: np.ndarray,
        method: str = 'morphological',
        iterations: int = 2
    ) -> np.ndarray:
        """Smooth boundaries of stitched mask.
        
        Args:
            mask: Binary mask
            method: Smoothing method ('morphological', 'gaussian', 'bilateral')
            iterations: Number of smoothing iterations
        
        Returns:
            Smoothed mask
        """
        if method == 'morphological':
            # Morphological closing followed by opening
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            smoothed = mask.copy()
            
            for _ in range(iterations):
                smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
                smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        
        elif method == 'gaussian':
            # Gaussian blur followed by thresholding
            smoothed = mask.astype(np.float32)
            
            for _ in range(iterations):
                smoothed = gaussian_filter(smoothed, sigma=1.5)
                smoothed = (smoothed > 0.5).astype(np.float32)
            
            smoothed = smoothed.astype(np.uint8)
        
        elif method == 'bilateral':
            # Bilateral filter for edge-preserving smoothing
            smoothed = mask.copy()
            
            for _ in range(iterations):
                smoothed = cv2.bilateralFilter(
                    smoothed.astype(np.float32),
                    d=9,
                    sigmaColor=75,
                    sigmaSpace=75
                )
                smoothed = (smoothed > 0.5).astype(np.uint8)
        
        else:
            smoothed = mask
        
        return smoothed
    
    @staticmethod
    def remove_seams(
        mask: np.ndarray,
        patch_boundaries: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Remove visible seams at patch boundaries.
        
        Args:
            mask: Stitched mask
            patch_boundaries: List of (row_start, row_end, col_start, col_end)
        
        Returns:
            Mask with seams removed
        """
        result = mask.copy()
        
        # Create seam mask
        seam_mask = np.zeros_like(mask)
        seam_width = 3
        
        for row_start, row_end, col_start, col_end in patch_boundaries:
            # Vertical seams
            if col_start > 0:
                seam_mask[row_start:row_end, 
                         max(0, col_start-seam_width):min(mask.shape[1], col_start+seam_width)] = 1
            
            # Horizontal seams
            if row_start > 0:
                seam_mask[max(0, row_start-seam_width):min(mask.shape[0], row_start+seam_width),
                         col_start:col_end] = 1
        
        # Inpaint seam regions
        if np.any(seam_mask):
            result = cv2.inpaint(
                result,
                seam_mask.astype(np.uint8),
                inpaintRadius=5,
                flags=cv2.INPAINT_TELEA
            )
        
        return result
    
    @staticmethod
    def connect_components(
        mask: np.ndarray,
        max_gap: int = 5
    ) -> np.ndarray:
        """Connect nearby components that might have been split.
        
        Args:
            mask: Binary mask
            max_gap: Maximum gap to bridge
        
        Returns:
            Mask with connected components
        """
        # Morphological closing with larger kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max_gap * 2 + 1, max_gap * 2 + 1)
        )
        
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Preserve original components and add connections
        result = np.maximum(mask, connected)
        
        return result


def create_quality_map(
    patches: List[Dict],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """Create a quality/confidence map for the stitched result.
    
    Args:
        patches: List of patches with confidence scores
        image_shape: Shape of full image
    
    Returns:
        Quality map with values 0-1
    """
    quality_map = np.zeros(image_shape, dtype=np.float32)
    coverage_map = np.zeros(image_shape, dtype=np.int32)
    
    for patch in patches:
        row, col = patch['row'], patch['col']
        h, w = patch['mask'].shape[:2] if 'mask' in patch else (patch['height'], patch['width'])
        confidence = patch.get('confidence', 1.0)
        
        end_row = min(row + h, image_shape[0])
        end_col = min(col + w, image_shape[1])
        
        quality_map[row:end_row, col:end_col] += confidence
        coverage_map[row:end_row, col:end_col] += 1
    
    # Normalize by coverage
    quality_map = np.divide(
        quality_map,
        coverage_map,
        out=np.zeros_like(quality_map),
        where=coverage_map > 0
    )
    
    return quality_map


def validate_stitching(
    stitched_mask: np.ndarray,
    patches: List[Dict],
    tolerance: float = 0.1
) -> Dict:
    """Validate stitching quality.
    
    Args:
        stitched_mask: Final stitched mask
        patches: Original patches
        tolerance: Acceptable difference threshold
    
    Returns:
        Validation report
    """
    report = {
        'total_patches': len(patches),
        'coverage': 0,
        'consistency_score': 0,
        'seam_artifacts': 0,
        'missing_regions': []
    }
    
    # Check coverage
    expected_coverage = np.zeros_like(stitched_mask)
    for patch in patches:
        row, col = patch['row'], patch['col']
        h, w = patch.get('height', 224), patch.get('width', 224)
        
        end_row = min(row + h, stitched_mask.shape[0])
        end_col = min(col + w, stitched_mask.shape[1])
        
        expected_coverage[row:end_row, col:end_col] = 1
    
    actual_coverage = (stitched_mask > 0).astype(np.float32)
    report['coverage'] = np.sum(actual_coverage) / np.sum(expected_coverage)
    
    # Check consistency in overlapping regions
    overlap_errors = []
    for i, patch1 in enumerate(patches):
        for patch2 in patches[i+1:]:
            # Check if patches overlap
            overlap = calculate_overlap(patch1, patch2)
            if overlap is not None:
                # Compare predictions in overlap region
                diff = compare_overlap_predictions(
                    stitched_mask,
                    patch1,
                    patch2,
                    overlap
                )
                if diff > tolerance:
                    overlap_errors.append(diff)
    
    if overlap_errors:
        report['consistency_score'] = 1.0 - np.mean(overlap_errors)
    else:
        report['consistency_score'] = 1.0
    
    # Detect seam artifacts (sharp transitions)
    gradients = np.gradient(stitched_mask.astype(np.float32))
    gradient_magnitude = np.sqrt(gradients[0]**2 + gradients[1]**2)
    report['seam_artifacts'] = np.sum(gradient_magnitude > 1.5)
    
    return report


def calculate_overlap(patch1: Dict, patch2: Dict) -> Optional[Tuple[int, int, int, int]]:
    """Calculate overlapping region between two patches.
    
    Returns:
        Tuple of (row_start, row_end, col_start, col_end) or None
    """
    r1, c1 = patch1['row'], patch1['col']
    h1, w1 = patch1.get('height', 224), patch1.get('width', 224)
    
    r2, c2 = patch2['row'], patch2['col']
    h2, w2 = patch2.get('height', 224), patch2.get('width', 224)
    
    # Calculate intersection
    row_start = max(r1, r2)
    row_end = min(r1 + h1, r2 + h2)
    col_start = max(c1, c2)
    col_end = min(c1 + w1, c2 + w2)
    
    if row_start < row_end and col_start < col_end:
        return (row_start, row_end, col_start, col_end)
    
    return None


def compare_overlap_predictions(
    stitched_mask: np.ndarray,
    patch1: Dict,
    patch2: Dict,
    overlap: Tuple[int, int, int, int]
) -> float:
    """Compare predictions in overlapping region."""
    row_start, row_end, col_start, col_end = overlap
    
    # Get overlap region from stitched mask
    stitched_region = stitched_mask[row_start:row_end, col_start:col_end]
    
    # Calculate difference metric
    diff = 0.0
    if 'mask' in patch1 and 'mask' in patch2:
        # Get corresponding regions from patches
        r1, c1 = patch1['row'], patch1['col']
        region1 = patch1['mask'][
            row_start-r1:row_end-r1,
            col_start-c1:col_end-c1
        ]
        
        r2, c2 = patch2['row'], patch2['col']
        region2 = patch2['mask'][
            row_start-r2:row_end-r2,
            col_start-c2:col_end-c2
        ]
        
        # Calculate difference
        diff = np.mean(np.abs(region1 - region2))
    
    return diff


if __name__ == "__main__":
    # Test stitching utilities
    print("Testing stitching utilities...")
    
    # Create synthetic patches
    image_shape = (512, 512)
    patch_size = 224
    overlap = 32
    
    stitcher = PatchStitcher(image_shape, patch_size, overlap, blend_mode='weighted')
    
    # Add some synthetic patches
    for row in range(0, 512-224+1, 192):
        for col in range(0, 512-224+1, 192):
            # Create synthetic mask
            mask = np.random.random((224, 224)) > 0.7
            stitcher.add_patch(mask.astype(np.float32), (row, col), confidence=0.9)
    
    # Get final mask
    final_mask = stitcher.get_final_mask()
    
    # Smooth boundaries
    smoothed = BoundaryRefinement.smooth_boundaries(final_mask, method='morphological')
    
    print(f"Stitched mask shape: {final_mask.shape}")
    print(f"Non-zero pixels: {np.sum(final_mask > 0)}")
    print("Test complete!")