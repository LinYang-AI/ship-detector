import os
import argparse
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import from training script
from scripts.train_vit import ViTShipClassifier, ShipPatchDataset, get_augmentation_transforms


def load_model(checkpoint_path: str, config: Dict) -> ViTShipClassifier:
    """Load trained model from checkpoint"""
    model = ViTShipClassifier(config)

    # Load weights
    if checkpoint_path.endswith('.ckpt'):
        # Pytorch Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Regular PyTorch checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    model.eval()
    return model


def evaluate_model(
    model: ViTShipClassifier,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect predictions.

    Returns:
        Tuple of (labels, predictions, probabilities)
    """
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, labels = batch
            images = images.to(device)

            # Forward pass
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict:
    """Compute classification metrics."""

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'roc_auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    }

    # Compute per-class metrics
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    metrics['true_negatives'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Negative predictive value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return metrics


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: str
):
    """Generate and save confusion matrix visualization."""

    cm = confusion_matrix(labels, preds)

    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        cmap='Blues',
        xticklabels=['No Ship', 'Ship'],
        yticklabels=['No Ship', 'Ship'],
        square=True,
        cbar_kws={'label': 'Counts'}
    )

    plt.title('Confusion Matrix - Ship Detection',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontisize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add percentage annotations
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(2):
        for j in range(2):
            percentage = cm_normalized[i, j] * 100
            text = f'\n({percentage:.1f}%)'
            plt.text(j + 0.5, i + 0.7, text, ha='center',
                     va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: str
):
    """Generate and save ROC curve."""

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2,
             linestyle='--', label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic - Ship Detection',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curve saved to {output_path}")


def plot_prediction_distribution(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: str,
):
    """Plot distribution of prediction probabilities"""

    plt.figure(figsize=(10, 6))

    # Separate probabilities by true class
    no_ship_probs = probs[labels == 0]
    ship_probs = probs[labels == 1]

    # Create histograms
    bins = np.linspace(0, 1, 31)
    plt.hist(no_ship_probs, bins=bins, alpha=0.6,
             label='No Ship (True)', color='blue', density=True)
    plt.hist(ship_probs, bins=bins, alpha=0.6,
             label='Ship (True)', color='red', density=True)

    # Add decision threshold line
    plt.axvline(x=0.5, color='black', linestyle='--',
                linewidth=2, label='Decision Threshold')

    plt.xlabel('Prediction Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Prediction Probabilities',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Prediction distribution saved to {output_path}")


def save_error_analysis(
    manifest_df: pd.DataFrame,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
    top_k: int = 20
):
    """Save analysis of misclassified samples"""

    # Add predictions to dataframe
    eval_df = manifest_df.copy()
    eval_df['true_label'] = labels.astype(int)
    eval_df['predictied_label'] = preds.astype(int)
    eval_df['prediction_prob'] = probs
    eval_df['correct'] = (labels == preds).astype(int)

    # Identify errors
    errors_df = eval_df[eval_df['correct'] == 0].copy()

    # Calculate confidence of errors
    errors_df['cofnidence'] = np.where(
        errors_df['predicted_label'] == 1,
        errors_df['prediction_prob'],
        1 - errors_df['prediction_prob']
    )

    # Sort by confidence (most confident errors first)
    errors_df = errors_df.sort_values('confidence', ascending=False)

    # Save top errors
    top_errors = errors_df.head(top_k)[
        ['patch_path', 'true_label', 'predicted_label', 'prediction_prob', 'confidence']]
    error_file = os.path.join(output_dir, 'top_errors.csv')
    top_errors.to_csv(error_file, index=False)

    print(f"Top {top_k} errors saved to {error_file}")

    # Error statistics
    stats = {
        'total_errors': len(errors_df),
        'false_positives': len(errors_df[errors_df['true_label'] == 0]),
        'false_negatives': len(errors_df[errors_df['true_label'] == 1]),
        'avg_error_confidence': errors_df['confidence'].mean(),
        'max_error_confidence': errors_df['confidence'].max(),
    }

    return stats


def main(
    checkpoint_path: str,
    config_path: str,
    manifest_path: str,
    output_dir: str,
    data_split: str = 'test',
):
    """Main evaluation function."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(manifest_path)

    # Split data (use same seed as training for consistency)
    if data_split == 'val':
        # Use validation split from training
        from sklearn.model_selection import train_test_split
        _, eval_df = train_test_split(
            df,
            test_size=config['data']['val_split'],
            random_state=config['data']['random_seed'],
            stratify=df['has_ship']
        )
    else:
        eval_df = df

    print(f"Evaluating on {len(eval_df)} samples")
    print(
        f"Ship patches: {eval_df['has_ship'].sum()}, ({eval_df['has_ship'].mean()*100:.1f}%)")

    # Create dataset and loader
    _, val_transform = get_augmentation_transforms(config)
    dataset = ShipPatchDatast(eval_df, transform=val_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=config['trainig']['batch_size'],
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
    labels, preds, probs = evaluate_model(model, dataloader, device)

    # Compute metrics
    metrics = compute_metrics(labels, preds, probs)

    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(
        f"  TN:  {metrics['true_negatives']}   FP:  {metrics['false_positives']}")
    print(
        f"  FN:  {metrics['false_negatives']}  TP:  {metrics['true_positives']}")
    print("="*50)

    # Generate visualization
    plot_confusion_matrix(labels, preds, os.path.join(
        output_dir, 'confusion_matrix.png'))
    plot_roc_curve(labels, probs, os.path.join('output_dir', 'roc_curve.png'))
    plot_prediction_distribution(
        labels, probs, os.path.join(output_dir, 'prediction_dist.png'))

    # Error analysis
    error_stats = save_error_analysis(
        eval_df, labels, preds, probs, output_dir)

    # Save metrics to JSON
    import json
    metrics['error_analysis'] = error_stats
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_file}")

    # Generate classification report
    report = classification_report(
        labels, preds,
        target_names=['No Ship', 'Ship'],
        output_dict=False
    )

    report_file = os.path.join(output_dir, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("SHIP DETECTION CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(report)
        f.write("\n" + "="*50 + "\n")
        f.write(f"\nTotal samples evaluated: {len(labels)}\n")
        f.write(
            f"Ship samples: {int(labels.sum())} ({labels.mean()*100:.1f}%)\n")
        f.write(
            f"No-ship samples: {int(len(labels) - labels.sum())} ({(1 - labels.mean())*100:.1f}%)\n")

    print(f"Classification report saved to {report_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ship detection classifier")
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to manifest CSV file')
    parser.add_argument('--output-dir', type=str,
                        default='./evaluation/vit', help='Path to output directory')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'], help='Data split to evaluate')

    args = parser.parse_args()

    metrics = main(
        args.checkpoint,
        args.config,
        args.manifest,
        args.output_dir,
        args.split
    )
