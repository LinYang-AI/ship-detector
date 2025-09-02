#!/usr/bin/env python3
"""
Memory profiling utilities for model training.
Helps identify memory bottlenecks and optimize settings.
"""

import os
import time
import tracemalloc
import psutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Callable
import gc


class MemoryProfiler:
    """Profile memory usage during training."""
    
    def __init__(self, log_file: str = "memory_profile.csv"):
        self.log_file = log_file
        self.memory_logs = []
        self.process = psutil.Process()
        tracemalloc.start()
    
    def profile(self, stage: str = "unknown") -> Dict:
        """Profile current memory usage."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process_info = self.process.memory_info()
        
        # Python memory
        current, peak = tracemalloc.get_traced_memory()
        
        # GPU memory if available
        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats = {
                'gpu_allocated_mb': torch.cuda.memory_allocated() / 1e6,
                'gpu_reserved_mb': torch.cuda.memory_reserved() / 1e6,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1e6,
            }
            torch.cuda.reset_peak_memory_stats()
        
        stats = {
            'timestamp': time.time(),
            'stage': stage,
            'system_ram_used_gb': memory.used / 1e9,
            'system_ram_available_gb': memory.available / 1e9,
            'system_ram_percent': memory.percent,
            'process_ram_gb': process_info.rss / 1e9,
            'process_vms_gb': process_info.vms / 1e9,
            'python_current_mb': current / 1e6,
            'python_peak_mb': peak / 1e6,
            **gpu_stats
        }
        
        self.memory_logs.append(stats)
        return stats
    
    def save_logs(self):
        """Save memory logs to CSV."""
        df = pd.DataFrame(self.memory_logs)
        df.to_csv(self.log_file, index=False)
        print(f"Memory profile saved to {self.log_file}")
    
    def plot_profile(self, output_path: str = "memory_profile.png"):
        """Plot memory usage over time."""
        df = pd.DataFrame(self.memory_logs)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # System RAM
        axes[0, 0].plot(df['timestamp'] - df['timestamp'].min(), 
                       df['system_ram_used_gb'], label='Used')
        axes[0, 0].plot(df['timestamp'] - df['timestamp'].min(),
                       df['system_ram_available_gb'], label='Available')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('GB')
        axes[0, 0].set_title('System RAM')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Process Memory
        axes[0, 1].plot(df['timestamp'] - df['timestamp'].min(),
                       df['process_ram_gb'], label='RSS')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('GB')
        axes[0, 1].set_title('Process Memory')
        axes[0, 1].grid(True)
        
        # Python Memory
        axes[1, 0].plot(df['timestamp'] - df['timestamp'].min(),
                       df['python_current_mb'], label='Current')
        axes[1, 0].plot(df['timestamp'] - df['timestamp'].min(),
                       df['python_peak_mb'], label='Peak')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('MB')
        axes[1, 0].set_title('Python Memory')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # GPU Memory (if available)
        if 'gpu_allocated_mb' in df.columns:
            axes[1, 1].plot(df['timestamp'] - df['timestamp'].min(),
                           df['gpu_allocated_mb'], label='Allocated')
            axes[1, 1].plot(df['timestamp'] - df['timestamp'].min(),
                           df['gpu_reserved_mb'], label='Reserved')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('MB')
            axes[1, 1].set_title('GPU Memory')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.suptitle('Memory Usage Profile')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Memory profile plot saved to {output_path}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        df = pd.DataFrame(self.memory_logs)
        
        summary = {
            'peak_system_ram_gb': df['system_ram_used_gb'].max(),
            'peak_process_ram_gb': df['process_ram_gb'].max(),
            'peak_python_mb': df['python_peak_mb'].max(),
            'avg_system_ram_percent': df['system_ram_percent'].mean(),
        }
        
        if 'gpu_allocated_mb' in df.columns:
            summary['peak_gpu_mb'] = df['gpu_max_allocated_mb'].max()
        
        return summary


def benchmark_data_loading(
    dataset,
    batch_size: int = 32,
    num_workers: int = 2,
    num_batches: int = 10
) -> Dict:
    """Benchmark data loading performance."""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    profiler = MemoryProfiler()
    profiler.profile("start")
    
    # Warm-up
    for i, batch in enumerate(loader):
        if i >= 2:
            break
    
    # Benchmark
    start_time = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        profiler.profile(f"batch_{i}")
    
    end_time = time.time()
    
    # Results
    results = {
        'total_time': end_time - start_time,
        'time_per_batch': (end_time - start_time) / num_batches,
        'samples_per_second': (batch_size * num_batches) / (end_time - start_time),
        **profiler.get_summary()
    }
    
    return results


def optimize_batch_size(
    model_class,
    config: Dict,
    max_batch_size: int = 128,
    device: str = 'cuda'
) -> int:
    """Find optimal batch size for given memory constraints."""
    
    print("Finding optimal batch size...")
    optimal_batch_size = 1
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        if batch_size > max_batch_size:
            break
        
        try:
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create model
            model = model_class(config).to(device)
            
            # Create dummy batch
            batch = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.get('precision', 16) == 16):
                output = model(batch)
                loss = output.mean()
            
            # Backward pass
            loss.backward()
            
            # If successful, this batch size works
            optimal_batch_size = batch_size
            print(f"  Batch size {batch_size}: ✓")
            
            # Cleanup
            del model, batch, output, loss
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  Batch size {batch_size}: ✗ (OOM)")
                break
            else:
                raise e
    
    print(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def compare_lora_vs_full(config_path: str):
    """Compare memory usage between LoRA and full fine-tuning."""
    import yaml
    from train_vit_efficient import EfficientViTClassifier
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results = {}
    
    # Test with LoRA
    config['model']['use_lora'] = True
    model_lora = EfficientViTClassifier(config)
    
    total_params_lora = sum(p.numel() for p in model_lora.parameters())
    trainable_params_lora = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    
    results['lora'] = {
        'total_params': total_params_lora,
        'trainable_params': trainable_params_lora,
        'reduction': f"{(1 - trainable_params_lora/total_params_lora)*100:.1f}%"
    }
    
    # Test without LoRA
    config['model']['use_lora'] = False
    model_full = EfficientViTClassifier(config)
    
    total_params_full = sum(p.numel() for p in model_full.parameters())
    trainable_params_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    
    results['full'] = {
        'total_params': total_params_full,
        'trainable_params': trainable_params_full,
        'reduction': "0.0%"
    }
    
    # Print comparison
    print("\n" + "="*60)
    print("LoRA vs Full Fine-tuning Comparison")
    print("="*60)
    print(f"Full Fine-tuning:")
    print(f"  Total Parameters: {results['full']['total_params']:,}")
    print(f"  Trainable Parameters: {results['full']['trainable_params']:,}")
    print(f"\nLoRA Fine-tuning:")
    print(f"  Total Parameters: {results['lora']['total_params']:,}")
    print(f"  Trainable Parameters: {results['lora']['trainable_params']:,}")
    print(f"  Parameter Reduction: {results['lora']['reduction']}")
    print(f"\nMemory Savings: ~{trainable_params_full/trainable_params_lora:.1f}x")
    print("="*60)
    
    return results


class DataLoaderOptimizer:
    """Optimize DataLoader settings for memory efficiency."""
    
    @staticmethod
    def find_optimal_workers(dataset, batch_size: int = 32) -> int:
        """Find optimal number of workers."""
        from torch.utils.data import DataLoader
        
        best_time = float('inf')
        best_workers = 0
        
        for num_workers in [0, 1, 2, 4, 8]:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True
            )
            
            # Benchmark
            start = time.time()
            for i, _ in enumerate(loader):
                if i >= 10:
                    break
            elapsed = time.time() - start
            
            print(f"Workers: {num_workers}, Time: {elapsed:.2f}s")
            
            if elapsed < best_time:
                best_time = elapsed
                best_workers = num_workers
        
        return best_workers


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory profiling utilities")
    parser.add_argument('--action', choices=['profile', 'optimize', 'compare'], 
                       default='compare', help='Action to perform')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    if args.action == 'compare' and args.config:
        compare_lora_vs_full(args.config)
    
    elif args.action == 'optimize':
        from train_vit_efficient import EfficientViTClassifier
        import yaml
        
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        optimal_batch = optimize_batch_size(EfficientViTClassifier, config)
        print(f"Recommended batch size: {optimal_batch}")
    
    elif args.action == 'profile':
        profiler = MemoryProfiler()
        
        # Simulate some operations
        for i in range(100):
            profiler.profile(f"step_{i}")
            time.sleep(0.1)
            
            # Simulate memory spike
            if i == 50:
                dummy = torch.randn(1000, 1000, 100)
                del dummy
        
        profiler.save_logs()
        profiler.plot_profile()
        
        summary = profiler.get_summary()
        print("\nMemory Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.2f}")