import os
from pathlib import Path
from typing import List, Optional, Dict
import logging
import time

import numpy as np
import pandas as pd
import cv2

from ultralytics import YOLO

from tqdm import tqdm
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLEProcessor:
    """Handle RLE encoding/decoding."""
    
    @ staticmethod
    def encode_rle(mask: np.ndarray) -> str:
        """Encode binary mask to RLE string."""
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        
        if len(runs) == 2 and runs[1] == len(pixels) - 1:
            return ''  # No mask
        
        return ' '.join(str(x) for x in runs)
    

class YOLOInference:
    """YOLO inference pipeline for ship segmentation."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: Path to YOLO model weights.
            device: Device for inference ('cuda' or 'cpu').
        """
        self.model = YOLO(model_path)
        self.device = device
        self.rle_processor = RLEProcessor()
        
        # Move model to device
        if device == 'cuda' and torch.cuda.is_available():
            self.model.to('cuda')
        else:
            self.model.to('cpu')
            self.device = 'cpu'
        
        logger.info(f"Model loaded on {self.device}")
        
    def process_image(self, image_path: str,
                      conf_thresh: float = 0.25,
                      iou_thresh: float = 0.45) -> Dict:
        """Process single image and return RLE predictions.
        
        Args:
            image_path: Path to input image
            conf_thresh: Confidence threshold for detections.
            iou_thresh: IoU threshold for NMS.
            
        Returns:
            Dictionary with image info and RLE masks.
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=640,
            retina_masks=True,
            classes=[0],
            verbose=False,
            save=False,
            stream=False
        )
        
        # Process results
        rle_masks = []
        boxes = []
        scores = []
        
        for r in results:
            if r.masks is not None:
                # Get original image shape
                orig_shape = r.orig_shape
                
                # Process masks
                masks = r.masks.data.cpu().numpy()
                
                # Process boxes and scores if available
                if r.boxes is not None:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    
                    for i, mask in enumerate(masks):
                        # Resize mask to original size
                        mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
                        mask = (mask > 0.5).astype(np.uint8)
                        
                        # Encode to RLE
                        rle = self.rle_processor.encode_rle(mask)
                        
                        if rle:  # Only add non-empty masks
                            rle_masks.append(rle)
                            
                            if i < len(boxes_xyxy):
                                boxes.append(boxes_xyxy[i].tolist())
                                scores.append(float(confs[i]))
                else:
                    # No boxes, just process masks
                    for mask in masks:
                        mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
                        mask = (mask > 0.5).astype(np.uint8)
                        
                        rle = self.rle_processor.encode_rle(mask)
                        if rle:
                            rle_masks.append(rle)
        
        return {
            'image_path': image_path,
            'image_id': Path(image_path).name,
            'rle_masks': rle_masks,
            'boxes': boxes,
            'scores': scores,
            'num_ships': len(rle_masks)
        }
        
    def process_directory(self, images_dir: str,
                          output_csv: str,
                          conf_thresh: float = 0.25,
                          iou_thresh: float = 0.45,
                          save_visualizations: bool = False):
        """Process all images in directory and save results to CSV.
        
        Args:
            images_dir: Directory containing images
            output_csv: Path to save predictions CSV
            conf_thresh: Confidence threshold for detections.
            iou_thresh: IoU threshold for NMS.
            save_visualizations: Whether to save visualization images.
        """
        images_dir = Path(images_dir)
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        logger.info(f"Found {len(image_files)} images in {images_dir} to process.")
        
        # Process images
        predictions = []
        total_time = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            start_time = time.time()
            
            # Process image
            result = self.process_image(
                str(img_path),
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Add predictions
            if result['rle_masks']:
                for rle in result['rle_masks']:
                    predictions.append({
                        'ImageId': result['image_id'],
                        'EncodedPixels': rle
                    })
            else:
                # No ships detected
                predictions.append({
                    'ImageId': result['image_id'],
                    'EncodedPixels': ''
                })
                
            # Save visualization if needed
            if save_visualizations and result['num_ships'] > 0:
                self._save_visualization(img_path, result)
        
        # Create DataFrame and save
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_csv, index=False)
        
        # Print statistics
        avg_time = total_time / len(image_files)
        fps = 1 / avg_time if avg_time > 0 else 0
        
        logger.info(f"\nProcessing complete!")
        logger.info(f"Total images: {len(image_files)}")
        logger.info(f"Total ships detected: {len([p for p in predictions if p['EncodedPixels']])}")
        logger.info(f"Average time per image: {avg_time:.4f} seconds")
        logger.info(f"Processing speed: {fps:.2f} FPS")
        logger.info(f"Predictions saved to: {output_csv}")
        
        return pred_df
    
    def _save_visualization(self, image_path: Path, result: Dict):
        """Save visualization of detection results."""
        # Create visualization of detection results
        vis_dir =  image_path.parent / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Load image
        image = cv2.imread(str(image_path))
        overlay = image.copy()
        
        # Decode and draw msks
        h, w = image.shape[:2]
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, rle in enumerate(result['rle_masks']):
            # Decode RLE
            mask =self._decode_rle_for_vis(rle, (h, w))
            
            # Apply color
            color = colors[i % len(colors)]
            overlay[mask == 1] = color
            
            # Draw box if available
            if i < len(result['boxes']):
                box = result['boxes'][i]
                score = result['scores'][i] if i < len(result['scores']) else 0
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"Ship {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Blend original image and overlay
        result_img = cv2.addWeighted(image, 0.5, overlay, 0.4, 0)
        
        # Save visualization
        vis_path =vis_dir / f"vis_{image_path.name}"
        cv2.imwrite(str(vis_path), result_img)
        
    def _decode_rle_for_vis(self, rle: str, shape: tuple) -> np.ndarray:
        """Decode RLE for visualization."""
        if not rle:
            return np.zeros(shape, dtype=np.uint8)
        
        s = rle.split()
        starts =np.array(s[0::2], dtype=int) - 1
        lengths = np.array(s[1::2], dtype=int)
        ends = starts + lengths
        
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, end in zip(starts, ends):
            img[start:end] = 1
        
        return img.reshape(shape, order='F')
    
    def benchmark(self, test_images: List[str], num_runs: int = 100):
        """Benchmark inference speed.
        
        Args
            test_images: List of test image paths
            num_runs: Number of inference runs
        """
        logger.info(f"Running benchmark with {num_runs} iteration...")
        
        # Warm-up
        for img in test_images[:min(5, len(test_images))]:
            _ =self.process_image(img)
            
        # Benchmark
        times = []
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            for img in test_images:
                start = time.time()
                _ = self.process_image(img)
                times.append(time.time() - start)
        
        times = np.array(times)
        
        logger.info(f"Benchmark Results:")
        logger.info(f"\tMean: {times.mean():.4f} seconds")
        logger.info(f"\t Std: {times.std():.4f} seconds")
        logger.info(f"\tMin: {times.min():.4f} seconds")
        logger.info(f"\tMax: {times.max():.4f} seconds")
        logger.info(f"\tFPS: {1 / times.mean():.2f}")


def create_submission_format(predictions_csv: str, output_csv: str):
    """Convert predictions to competition submission format.
    
    Args:
        predictions_csv: Path to predictions CSV
        output_csv: Path to save submission CSV
    """
    # Load predictions
    pred_df =pd.read_csv(predictions_csv)
    
    # Ensure correct format
    submission_df = pred_df[['ImageId', 'EncodedPixels']].copy()
    
    # Fill NaN with empty strings
    submission_df['EncodedPixels'] = submission_df['EncodedPixels'].fillna('')
    
    # Save
    submission_df.to_csv(output_csv, index=False)
    logger.info(f"Submission file saved to {output_csv}")
    
    # Print statistics
    total_images = submission_df['ImageId'].nunique()
    images_with_ships = len(submission_df[submission_df['EncodedPixels'] != ''])
    
    logger.info(f"Total unique images: {total_images}")
    logger.info(f"Images with ships: {images_with_ships}")
    logger.info(f"Total ship instances: {len(submission_df)}")
