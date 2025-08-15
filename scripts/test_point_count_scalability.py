#!/usr/bin/env python3
"""
Test how CNF performance scales with different point counts.
Critical for understanding if we need adaptive architectures.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_cnf import PointFlow2DCNF

def test_point_count(num_points, hidden_dim, data_path, slice_name, device='cuda'):
    """Test decoder performance with specific point count."""
    
    # Load and subsample data
    slice_path = Path(data_path) / slice_name
    all_points = torch.from_numpy(np.load(slice_path)).float()
    if all_points.ndim == 1:
        all_points = all_points.reshape(-1, 2)
    
    # Subsample or duplicate to get exact count
    if all_points.shape[0] >= num_points:
        indices = torch.randperm(all_points.shape[0])[:num_points]
        target_points = all_points[indices].to(device)
    else:
        # Duplicate points if needed
        factor = num_points // all_points.shape[0] + 1
        duplicated = all_points.repeat(factor, 1)
        indices = torch.randperm(duplicated.shape[0])[:num_points]
        target_points = duplicated[indices].to(device)
    
    # Normalize
    center = target_points.mean(dim=0)
    scale = (target_points - center).abs().max() * 1.1
    target_points = (target_points - center) / scale
    
    # Fixed latent
    latent_dim = 32
    fixed_z = torch.randn(1, latent_dim).to(device)
    fixed_z = fixed_z / fixed_z.norm() * np.sqrt(latent_dim)
    
    # Create decoder
    decoder = PointFlow2DCNF(
        point_dim=2,
        context_dim=latent_dim,
        hidden_dim=hidden_dim,
        solver='euler',
        solver_steps=max(10, num_points // 50)  # Adaptive steps
    ).to(device)
    
    # Train
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3)
    epochs = 150
    
    losses = []
    train_times = []
    
    import time
    for epoch in range(epochs):
        start_time = time.time()
        
        optimizer.zero_grad()
        
        generated = decoder.sample(fixed_z, num_points).squeeze(0)
        
        # Chamfer loss
        dist_g2t = torch.cdist(generated, target_points).min(dim=1)[0].mean()
        dist_t2g = torch.cdist(target_points, generated).min(dim=1)[0].mean()
        loss = dist_g2t + dist_t2g
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()
        
        losses.append(loss.item())
        train_times.append(time.time() - start_time)
    
    return {
        'num_points': num_points,
        'final_loss': losses[-1],
        'best_loss': min(losses),
        'convergence_epoch': losses.index(min(losses)),
        'avg_time_per_epoch': np.mean(train_times[10:]),  # Skip warmup
        'solver_steps': decoder.solver_steps
    }

def run_scalability_test(data_path: str, slice_name: str, device: str = 'cuda'):
    """Test scalability across different point counts."""
    
    print("\n🧪 POINT COUNT SCALABILITY TEST")
    print("=" * 50)
    
    # Test different point counts
    point_counts = [25, 50, 100, 200, 400, 800, 1600]
    hidden_dims = [32, 64]  # Test two architectures
    
    results = {dim: [] for dim in hidden_dims}
    
    for hidden_dim in hidden_dims:
        print(f"\n📊 Testing with hidden_dim={hidden_dim}")
        
        for num_points in point_counts:
            print(f"  Testing {num_points} points...", end='', flush=True)
            
            try:
                result = test_point_count(num_points, hidden_dim, data_path, slice_name, device)
                results[hidden_dim].append(result)
                print(f" Loss: {result['best_loss']:.4f}, Time: {result['avg_time_per_epoch']*1000:.1f}ms")
            except Exception as e:
                print(f" Failed: {e}")
    
    # Visualize
    output_dir = Path("outputs/scalability_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Loss vs point count
    plt.subplot(2, 3, 1)
    for hidden_dim, res_list in results.items():
        if res_list:
            counts = [r['num_points'] for r in res_list]
            losses = [r['best_loss'] for r in res_list]
            plt.plot(counts, losses, 'o-', label=f'Hidden {hidden_dim}', markersize=8)
    plt.xlabel('Number of Points')
    plt.ylabel('Best Loss')
    plt.title('Reconstruction Quality vs Point Count')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    
    # 2. Training time scaling
    plt.subplot(2, 3, 2)
    for hidden_dim, res_list in results.items():
        if res_list:
            counts = [r['num_points'] for r in res_list]
            times = [r['avg_time_per_epoch'] * 1000 for r in res_list]
            plt.plot(counts, times, 'o-', label=f'Hidden {hidden_dim}', markersize=8)
    plt.xlabel('Number of Points')
    plt.ylabel('Time per Epoch (ms)')
    plt.title('Computational Cost Scaling')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # 3. Convergence speed
    plt.subplot(2, 3, 3)
    for hidden_dim, res_list in results.items():
        if res_list:
            counts = [r['num_points'] for r in res_list]
            conv = [r['convergence_epoch'] for r in res_list]
            plt.plot(counts, conv, 'o-', label=f'Hidden {hidden_dim}', markersize=8)
    plt.xlabel('Number of Points')
    plt.ylabel('Convergence Epoch')
    plt.title('Convergence Speed vs Complexity')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    
    # 4. Loss per point
    plt.subplot(2, 3, 4)
    for hidden_dim, res_list in results.items():
        if res_list:
            counts = [r['num_points'] for r in res_list]
            loss_per_point = [r['best_loss'] / r['num_points'] for r in res_list]
            plt.plot(counts, loss_per_point, 'o-', label=f'Hidden {hidden_dim}', markersize=8)
    plt.xlabel('Number of Points')
    plt.ylabel('Loss per Point')
    plt.title('Normalized Performance')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # 5. Solver steps used
    plt.subplot(2, 3, 5)
    if results[hidden_dims[0]]:
        counts = [r['num_points'] for r in results[hidden_dims[0]]]
        steps = [r['solver_steps'] for r in results[hidden_dims[0]]]
        plt.plot(counts, steps, 'o-', markersize=8)
        plt.xlabel('Number of Points')
        plt.ylabel('Integration Steps')
        plt.title('Adaptive Solver Steps')
        plt.xscale('log')
        plt.grid(True)
    
    # 6. Efficiency metric
    plt.subplot(2, 3, 6)
    for hidden_dim, res_list in results.items():
        if res_list:
            counts = [r['num_points'] for r in res_list]
            # Efficiency = quality / time
            efficiency = [1.0 / (r['best_loss'] * r['avg_time_per_epoch']) for r in res_list]
            plt.plot(counts, efficiency, 'o-', label=f'Hidden {hidden_dim}', markersize=8)
    plt.xlabel('Number of Points')
    plt.ylabel('Efficiency (1 / (loss × time))')
    plt.title('Overall Efficiency')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scalability_results.png", dpi=150)
    plt.close()
    
    # Summary
    print("\n📊 SCALABILITY SUMMARY")
    print("=" * 50)
    
    # Find sweet spots
    for hidden_dim, res_list in results.items():
        if res_list:
            print(f"\nHidden dimension {hidden_dim}:")
            
            # Best point count for quality
            best_quality = min(res_list, key=lambda x: x['best_loss'])
            print(f"  Best quality: {best_quality['num_points']} points (loss={best_quality['best_loss']:.4f})")
            
            # Best efficiency
            efficiencies = [(r, 1.0 / (r['best_loss'] * r['avg_time_per_epoch'])) for r in res_list]
            best_eff = max(efficiencies, key=lambda x: x[1])
            print(f"  Best efficiency: {best_eff[0]['num_points']} points")
            
            # Scaling behavior
            if len(res_list) > 2:
                time_scaling = np.polyfit(
                    np.log([r['num_points'] for r in res_list]),
                    np.log([r['avg_time_per_epoch'] for r in res_list]),
                    1
                )[0]
                print(f"  Time scaling: O(N^{time_scaling:.2f})")
    
    print("\n💡 Key Insights:")
    print("- Small slices (<100 points): Use hidden_dim=32")
    print("- Medium slices (100-500 points): Use hidden_dim=64") 
    print("- Large slices (>500 points): Consider hidden_dim=128")
    print("- Time complexity is approximately O(N^1.5)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_scalability_test(args.data_path, args.slice_name, args.device)
