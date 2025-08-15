#!/usr/bin/env python3
"""
Test memory efficiency and batch processing capabilities.
Important for scaling to full dataset training.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import argparse

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.encoder import PointNet2DEncoder
from models.pointflow2d_cnf import PointFlow2DCNF

def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def test_batch_processing(data_path: str, slice_name: str, device: str = 'cuda'):
    """Test how model handles different batch sizes."""
    
    print("\n🧪 BATCH PROCESSING & MEMORY TEST")
    print("=" * 50)
    
    # Load data
    slice_path = Path(data_path) / slice_name
    points = torch.from_numpy(np.load(slice_path)).float().to(device)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    
    # Normalize
    center = points.mean(dim=0)
    scale = (points - center).abs().max() * 1.1
    points = (points - center) / scale
    num_points = points.shape[0]
    
    # Create model
    encoder = PointNet2DEncoder(input_dim=2, latent_dim=32, hidden_dim=128).to(device)
    decoder = PointFlow2DCNF(point_dim=2, context_dim=32, hidden_dim=64, 
                            solver='euler', solver_steps=20).to(device)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing batch size {batch_size}")
        
        # Create batch
        batch_points = points.unsqueeze(0).repeat(batch_size, 1, 1)
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        initial_memory = get_memory_usage()
        
        try:
            # Forward pass
            start_time = time.time()
            
            # Encode
            mu, logvar = encoder(batch_points)
            z = mu  # Deterministic for testing
            
            # Decode
            reconstructed = []
            for i in range(batch_size):
                recon = decoder.sample(z[i:i+1], num_points)
                reconstructed.append(recon)
            reconstructed = torch.cat(reconstructed, dim=0)
            
            forward_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            memory_per_sample = (peak_memory - initial_memory) / batch_size
            
            # Backward pass
            loss = ((reconstructed - batch_points) ** 2).mean()
            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time
            
            results.append({
                'batch_size': batch_size,
                'forward_time': forward_time,
                'backward_time': backward_time,
                'total_time': forward_time + backward_time,
                'time_per_sample': (forward_time + backward_time) / batch_size,
                'peak_memory': peak_memory,
                'memory_per_sample': memory_per_sample
            })
            
            print(f"   Forward: {forward_time*1000:.1f}ms, Backward: {backward_time*1000:.1f}ms")
            print(f"   Memory: {peak_memory:.1f}MB total, {memory_per_sample:.1f}MB/sample")
            
        except RuntimeError as e:
            print(f"   Failed: {e}")
            break
    
    # Visualize results
    if results:
        output_dir = Path("outputs/memory_efficiency")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # Time scaling
        plt.subplot(2, 2, 1)
        batch_sizes_plot = [r['batch_size'] for r in results]
        times_per_sample = [r['time_per_sample'] * 1000 for r in results]
        plt.plot(batch_sizes_plot, times_per_sample, 'o-', markersize=10)
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Sample (ms)')
        plt.title('Processing Efficiency')
        plt.grid(True)
        
        # Memory scaling
        plt.subplot(2, 2, 2)
        memory_per_sample = [r['memory_per_sample'] for r in results]
        plt.plot(batch_sizes_plot, memory_per_sample, 's-', markersize=10)
        plt.xlabel('Batch Size')
        plt.ylabel('Memory per Sample (MB)')
        plt.title('Memory Efficiency')
        plt.grid(True)
        
        # Total time
        plt.subplot(2, 2, 3)
        total_times = [r['total_time'] * 1000 for r in results]
        plt.bar(batch_sizes_plot, total_times)
        plt.xlabel('Batch Size')
        plt.ylabel('Total Time (ms)')
        plt.title('Batch Processing Time')
        
        # Speedup
        plt.subplot(2, 2, 4)
        single_sample_time = results[0]['total_time']
        speedups = [single_sample_time * r['batch_size'] / r['total_time'] for r in results]
        plt.plot(batch_sizes_plot, speedups, 'd-', markersize=10)
        plt.plot(batch_sizes_plot, batch_sizes_plot, '--', alpha=0.5, label='Ideal')
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup')
        plt.title('Batch Processing Speedup')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "batch_efficiency.png", dpi=150)
        plt.close()
    
    # Test maximum batch size
    print("\n📊 Finding maximum batch size...")
    max_batch = 1
    while max_batch < 256:
        try:
            test_batch = points.unsqueeze(0).repeat(max_batch * 2, 1, 1)
            mu, _ = encoder(test_batch)
            del test_batch, mu
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            max_batch *= 2
        except RuntimeError:
            break
    
    print(f"Maximum batch size: {max_batch}")
    
    # Memory usage breakdown
    print("\n📊 Memory Usage Breakdown:")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()) * 4 / 1024 / 1024:.2f} MB")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()) * 4 / 1024 / 1024:.2f} MB")
    
    return results

def test_generation_speed(data_path: str, slice_name: str, device: str = 'cuda'):
    """Test generation speed for deployment."""
    
    print("\n🧪 GENERATION SPEED TEST")
    print("=" * 50)
    
    # Different point counts to generate
    point_counts = [100, 500, 1000, 2000]
    
    # Create decoder
    decoder = PointFlow2DCNF(
        point_dim=2,
        context_dim=32,
        hidden_dim=64,
        solver='euler',
        solver_steps=10  # Fewer steps for speed
    ).to(device)
    
    # Fixed latent
    z = torch.randn(1, 32).to(device)
    
    results = []
    
    for num_points in point_counts:
        # Warmup
        for _ in range(5):
            _ = decoder.sample(z, num_points)
        
        # Time generation
        times = []
        for _ in range(50):
            start = time.time()
            generated = decoder.sample(z, num_points)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        
        print(f"{num_points} points: {avg_time:.2f} ± {std_time:.2f} ms")
        results.append({
            'points': num_points,
            'avg_time': avg_time,
            'std_time': std_time
        })
    
    # FPS calculation for real-time generation
    print("\n📊 Real-time Generation Capability:")
    for res in results:
        fps = 1000 / res['avg_time']
        print(f"{res['points']} points: {fps:.1f} FPS")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Run tests
    batch_results = test_batch_processing(args.data_path, args.slice_name, args.device)
    gen_results = test_generation_speed(args.data_path, args.slice_name, args.device)
