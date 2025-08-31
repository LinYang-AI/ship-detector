import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple, List
from pathlib import Path


def create_detection_overlay(
    image: np.ndarray,
    has_ship: bool,
    confidence: float = 1.0,
    thickness: int = 3
) -> np.ndarray:
    """Add detection result overlay to patch image.
    
    Args:
        image: Input image (H, W, 3)
        has_ship: Detection result
        confidence: Detection confidence score
        thickness: Border thickness
    
    Returns:
        Image with detection overlay
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    
    # Choose color based on detection
    color = (0, 255, 0) if has_ship else (255, 0, 0)  # Green for ship, red for no ship
    label = f"Ship: {confidence:.2f}" if has_ship else f"No Ship: {confidence:.2f}"
    
    # Draw border
    cv2.rectangle(overlay, (0, 0), (w-1, h-1), color, thickness)
    
    # Add label with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(overlay, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
    cv2.putText(overlay, label, (10, text_h + 10), font, font_scale, color, font_thickness)
    
    return overlay


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Create segmentation mask overlay on image.
    
    Args:
        image: Original image (H, W, 3)
        mask: Binary mask (H, W)
        alpha: Overlay transparency
        color: Mask color (B, G, R)
    
    Returns:
        Image with mask overlay
    """
    # Ensure correct dimensions
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    
    # Ensure binary
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    
    # Apply mask
    masked = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_binary)
    
    # Blend with original
    result = cv2.addWeighted(image, 1-alpha, masked, alpha, 0)
    
    # Add contours for clarity
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    
    return result


def create_comparison_grid(
    image: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    title: str = "Ship Segmentation"
) -> np.ndarray:
    """Create a comparison grid showing image, ground truth, and prediction.
    
    Args:
        image: Original image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        title: Title for the grid
    
    Returns:
        Combined visualization grid
    """
    # Determine grid size
    num_imgs = 1 + (gt_mask is not None) + (pred_mask is not None)
    
    fig, axes = plt.subplots(1, num_imgs, figsize=(5*num_imgs, 5))
    if num_imgs == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    idx = 1
    
    # Ground truth
    if gt_mask is not None:
        overlay_gt = create_segmentation_overlay(image, gt_mask, color=(0, 255, 0))
        axes[idx].imshow(overlay_gt)
        axes[idx].set_title("Ground Truth")
        axes[idx].axis('off')
        idx += 1
    
    # Prediction
    if pred_mask is not None:
        overlay_pred = create_segmentation_overlay(image, pred_mask, color=(255, 0, 0))
        axes[idx].imshow(overlay_pred)
        axes[idx].set_title("Prediction")
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to image
    fig.canvas.draw()
    grid_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_image = grid_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return grid_image


def visualize_instance_segmentation(
    image: np.ndarray,
    masks: List[np.ndarray],
    scores: Optional[List[float]] = None,
    min_score: float = 0.5
) -> np.ndarray:
    """Visualize multiple ship instances with different colors.
    
    Args:
        image: Original image
        masks: List of binary masks for each instance
        scores: Confidence scores for each instance
        min_score: Minimum score threshold
    
    Returns:
        Image with colored instance masks
    """
    result = image.copy()
    
    # Color palette for instances
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for idx, mask in enumerate(masks):
        # Skip low confidence masks
        if scores and scores[idx] < min_score:
            continue
        
        # Get color for this instance
        color = colors[idx % len(colors)]
        
        # Apply mask
        mask_binary = (mask > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[:, :] = color
        masked = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_binary)
        
        # Blend
        result = cv2.addWeighted(result, 1, masked, 0.3, 0)
        
        # Add contour
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
        
        # Add score label if available
        if scores and len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, f"{scores[idx]:.2f}", (cx-20, cy), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result


def plot_training_history(
    history_file: str,
    output_path: str,
    metrics: List[str] = ['loss', 'iou', 'dice']
):
    """Plot training history from logs.
    
    Args:
        history_file: Path to CSV with training history
        output_path: Output path for plot
        metrics: Metrics to plot
    """
    import pandas as pd
    
    # Load history
    df = pd.read_csv(history_file)
    
    # Create subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training and validation
        if f'train_{metric}' in df.columns:
            ax.plot(df['epoch'], df[f'train_{metric}'], label='Train', linewidth=2)
        if f'val_{metric}' in df.columns:
            ax.plot(df['epoch'], df[f'val_{metric}'], label='Validation', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {output_path}")


def create_error_heatmap(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    image_shape: Tuple[int, int],
    output_path: str
):
    """Create heatmap showing common error locations.
    
    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        image_shape: Shape for the heatmap
        output_path: Path to save heatmap
    """
    # Accumulate errors
    error_map = np.zeros(image_shape, dtype=np.float32)
    
    for pred, gt in zip(pred_masks, gt_masks):
        # Resize if needed
        if pred.shape != image_shape:
            pred = cv2.resize(pred, (image_shape[1], image_shape[0]))
            gt = cv2.resize(gt, (image_shape[1], image_shape[0]))
        
        # Calculate error
        pred_binary = (pred > 0.5).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)
        error = np.abs(pred_binary - gt_binary)
        error_map += error
    
    # Normalize
    if len(pred_masks) > 0:
        error_map /= len(pred_masks)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Error Frequency')
    plt.title('Segmentation Error Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # Add grid
    plt.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Error heatmap saved to {output_path}")


if __name__ == "__main__":
    # Test visualizations with synthetic data
    print("Testing visualization utilities...")
    
    # Create synthetic image and mask
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[50:150, 50:150] = 255  # Square ship
    
    # Test detection overlay
    det_overlay = create_detection_overlay(image, True, 0.95)
    cv2.imwrite("test_detection_overlay.png", det_overlay)
    
    # Test segmentation overlay
    seg_overlay = create_segmentation_overlay(image, mask)
    cv2.imwrite("test_segmentation_overlay.png", seg_overlay)
    
    # Test comparison grid
    grid = create_comparison_grid(image, mask, mask)
    cv2.imwrite("test_comparison_grid.png", grid)
    
    print("Test visualizations saved!")