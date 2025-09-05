#!/usr/bin/env python3
"""
Evaluate U-Net segmentation model and generate metrics.
Computes IoU, Dice per image and creates visualizations.
"""

import os
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import from training script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_unet import (
    UNetShipSegmentation,
    ShipSegmentationDataset,
    get_augmentation_transforms
)


def load_model(checkpoint_path: str, config: Dict) -> UNetShipSegmentation:
    """Load trained model from checkpoint."""
    model = UNetShipSegmentation(config)
    
    if checkpoint_path.endswith('.ckpt'):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    
    model.eval()
    return model


def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Calculate segmentation metrics for a single image.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
    
    Returns:
        Dictionary of metrics
    """
    # Ensure binary
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = pred_sum + target_sum - intersection
    
    # IoU
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice
    dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    
    # Pixel accuracy
    correct = (pred == target).sum()
    total = pred.size
    pixel_acc = correct / total
    
    # True positive, false positive, false negative
    tp = intersection
    fp = pred_sum - intersection
    fn = target_sum - intersection
    tn = total - tp - fp - fn
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'pixel_acc': float(pixel_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': float(tp),
        'fp': float(fp),
        'fn': float(fn),
        'tn': float(tn)
    }


def evaluate_model(
    model: UNetShipSegmentation,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_predictions: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray]]:
    """Evaluate model on dataset.
    
    Returns:
        Tuple of (metrics_list, predictions, targets)
    """
    model = model.to(device)
    model.eval()
    
    all_metrics = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # Convert to numpy
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Calculate metrics for each image in batch
            for i in range(len(images)):
                pred_mask = preds_np[i, 0]  # Remove channel dimension
                target_mask = masks_np[i, 0]
                
                metrics = calculate_metrics(pred_mask, target_mask)
                all_metrics.append(metrics)
                
                if save_predictions and output_dir and batch_idx < 5:  # Save first 5 batches
                    all_predictions.append(pred_mask)
                    all_targets.append(target_mask)
    
    return all_metrics, all_predictions, all_targets


def create_overlay_visualization(
    image: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """Create overlay visualization of predictions.
    
    Args:
        image: Original RGB image (H, W, 3)
        pred_mask: Predicted mask (H, W)
        target_mask: Ground truth mask (H, W)
        alpha: Overlay transparency
    
    Returns:
        Overlay image with colored masks
    """
    # Ensure image is RGB and normalized
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create colored overlays
    overlay = image.copy()
    
    # Ground truth in green
    gt_mask = (target_mask > 0.5).astype(np.uint8)
    overlay[:, :, 1] = np.where(gt_mask, 
                                np.minimum(255, overlay[:, :, 1] + 100),
                                overlay[:, :, 1])
    
    # Predictions in red
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
    overlay[:, :, 0] = np.where(pred_mask_binary,
                                np.minimum(255, overlay[:, :, 0] + 100),
                                overlay[:, :, 0])
    
    # Blend with original
    result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    # Add legend
    result = cv2.putText(result, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    result = cv2.putText(result, "Pred", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return result


def save_visualizations(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    output_dir: str,
    num_samples: int = 10
):
    """Save visualization grids of predictions."""
    
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    n = min(num_samples, len(predictions))
    
    # Create grid of predictions
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        # Ground truth
        axes[i, 0].imshow(targets[i], cmap='gray')
        axes[i, 0].set_title(f'Ground Truth {i+1}')
        axes[i, 0].axis('off')
        
        # Prediction
        axes[i, 1].imshow(predictions[i], cmap='gray')
        axes[i, 1].set_title(f'Prediction {i+1}')
        axes[i, 1].axis('off')
        
        # Difference
        diff = np.abs(targets[i] - predictions[i])
        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f'Difference {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'predictions_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization grid to {vis_dir / 'predictions_grid.png'}")


def plot_metrics_distribution(metrics_list: List[Dict], output_dir: str):
    """Plot distribution of metrics."""
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    metrics_to_plot = ['iou', 'dice', 'pixel_acc', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Histogram
        ax.hist(df[metric], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(df[metric].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {df[metric].mean():.3f}')
        ax.axvline(df[metric].median(), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {df[metric].median():.3f}')
        
        ax.set_xlabel(metric.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{metric.upper()} Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Segmentation Metrics Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'metrics_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics distribution to {output_path}")


def export_to_geojson(
    mask: np.ndarray,
    transform: Optional[Dict] = None,
    crs: Optional[str] = None,
    simplify_tolerance: float = 1.0
) -> Dict:
    """Convert binary mask to GeoJSON format.
    
    Args:
        mask: Binary mask (H, W)
        transform: Geo transform (from rasterio)
        crs: Coordinate reference system
        simplify_tolerance: Douglas-Peucker simplification tolerance
    
    Returns:
        GeoJSON dictionary
    """
    from shapely.geometry import shape, mapping, Polygon
    from shapely.ops import unary_union
    import rasterio.features
    
    # Ensure binary
    mask = (mask > 0.5).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    for contour in contours:
        if len(contour) < 3:  # Need at least 3 points for polygon
            continue
        
        # Convert to polygon coordinates
        coords = contour.squeeze().tolist()
        if len(coords) < 3:
            continue
        
        # Close polygon if not closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        # Create polygon
        try:
            poly = Polygon(coords)
            
            # Simplify if requested
            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance)
            
            # Apply geo transform if available
            if transform:
                # Transform pixel coordinates to geo coordinates
                from rasterio.transform import xy
                geo_coords = []
                for x, y in poly.exterior.coords:
                    lon, lat = xy(transform, y, x)  # Note: row, col order
                    geo_coords.append([lon, lat])
                poly = Polygon(geo_coords)
            
            # Create feature
            feature = {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "class": "ship",
                    "area": poly.area,
                    "perimeter": poly.length
                }
            }
            features.append(feature)
        except Exception as e:
            print(f"Warning: Failed to create polygon: {e}")
            continue
    
    # Create GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    if crs:
        geojson["crs"] = {
            "type": "name",
            "properties": {"name": crs}
        }
    
    return geojson


def analyze_errors(metrics_list: List[Dict], output_dir: str):
    """Analyze and report error patterns."""
    
    df = pd.DataFrame(metrics_list)
    
    # Identify poor predictions
    poor_threshold = df['iou'].quantile(0.25)
    poor_preds = df[df['iou'] < poor_threshold]
    
    # Error analysis report
    report = {
        'total_samples': len(df),
        'mean_iou': float(df['iou'].mean()),
        'std_iou': float(df['iou'].std()),
        'mean_dice': float(df['dice'].mean()),
        'std_dice': float(df['dice'].std()),
        'poor_predictions': len(poor_preds),
        'poor_threshold': float(poor_threshold),
        'failure_rate': len(df[df['iou'] < 0.5]) / len(df),
        'perfect_predictions': len(df[df['iou'] > 0.95]),
        'quartiles': {
            'q1': float(df['iou'].quantile(0.25)),
            'median': float(df['iou'].median()),
            'q3': float(df['iou'].quantile(0.75))
        }
    }
    
    # Save error analysis
    error_file = os.path.join(output_dir, 'error_analysis.json')
    with open(error_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Error analysis saved to {error_file}")
    return report


def main(
    checkpoint_path: str,
    config_path: str,
    manifest_path: str,
    output_dir: str,
    num_samples: int = 100,
    save_geojson: bool = False
):
    """Main evaluation function."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(manifest_path)
    ship_df = df[df['has_ship'] == 1].copy()
    
    if len(ship_df) == 0:
        print("No ship patches found!")
        return
    
    # Split data (use same seed as training)
    _, val_df = train_test_split(
        ship_df,
        test_size=config['data']['val_split'],
        random_state=config['data']['random_seed']
    )
    
    # Limit samples if specified
    if num_samples > 0 and len(val_df) > num_samples:
        val_df = val_df.sample(n=num_samples, random_state=42)
    
    print(f"Evaluating on {len(val_df)} samples")
    
    # Create dataset and loader
    _, val_transform = get_augmentation_transforms(config)
    dataset = ShipSegmentationDataset(
        val_df,
        transform=val_transform,
        only_ship_patches=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Evaluate
    metrics_list, predictions, targets = evaluate_model(
        model, dataloader, device,
        save_predictions=True,
        output_dir=output_dir
    )
    
    # Aggregate metrics
    df_metrics = pd.DataFrame(metrics_list)
    
    # Print summary
    print("\n" + "="*60)
    print("SEGMENTATION EVALUATION RESULTS")
    print("="*60)
    print(f"Samples evaluated: {len(metrics_list)}")
    print(f"\nMean Metrics:")
    print(f"  IoU:        {df_metrics['iou'].mean():.4f} ± {df_metrics['iou'].std():.4f}")
    print(f"  Dice:       {df_metrics['dice'].mean():.4f} ± {df_metrics['dice'].std():.4f}")
    print(f"  Pixel Acc:  {df_metrics['pixel_acc'].mean():.4f} ± {df_metrics['pixel_acc'].std():.4f}")
    print(f"  Precision:  {df_metrics['precision'].mean():.4f} ± {df_metrics['precision'].std():.4f}")
    print(f"  Recall:     {df_metrics['recall'].mean():.4f} ± {df_metrics['recall'].std():.4f}")
    print(f"  F1:         {df_metrics['f1'].mean():.4f} ± {df_metrics['f1'].std():.4f}")
    print("="*60)
    
    # Save detailed metrics
    metrics_file = os.path.join(output_dir, 'metrics_detailed.csv')
    df_metrics.to_csv(metrics_file, index=False)
    print(f"\nDetailed metrics saved to {metrics_file}")
    
    # Save summary
    summary = {
        'num_samples': len(metrics_list),
        'mean_metrics': df_metrics.mean().to_dict(),
        'std_metrics': df_metrics.std().to_dict(),
        'min_metrics': df_metrics.min().to_dict(),
        'max_metrics': df_metrics.max().to_dict()
    }
    
    summary_file = os.path.join(output_dir, 'metrics_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate visualizations
    if predictions:
        save_visualizations(predictions, targets, output_dir)
    
    plot_metrics_distribution(metrics_list, output_dir)
    
    # Error analysis
    error_report = analyze_errors(metrics_list, output_dir)
    
    summary['error_report'] = error_report
    # Export sample to GeoJSON if requested
    if save_geojson and predictions:
        geojson_dir = Path(output_dir) / 'geojson'
        geojson_dir.mkdir(exist_ok=True)
        
        for i in range(min(5, len(predictions))):
            geojson = export_to_geojson(predictions[i])
            geojson_file = geojson_dir / f'prediction_{i:03d}.geojson'
            
            import json
            with open(geojson_file, 'w') as f:
                json.dump(geojson, f, indent=2)
        
        print(f"Sample GeoJSON files saved to {geojson_dir}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--manifest', type=str, required=True, help='Path to data manifest CSV')
    parser.add_argument('--output-dir', type=str, default='./evaluation/unet', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to evaluate (0=all)')
    parser.add_argument('--save-geojson', action='store_true', help='Export sample predictions to GeoJSON')
    
    args = parser.parse_args()
    
    summary = main(
        args.checkpoint,
        args.config,
        args.manifest,
        args.output_dir,
        args.num_samples,
        args.save_geojson
    )