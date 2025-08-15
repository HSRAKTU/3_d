#!/usr/bin/env python3
"""
CNF Architecture Ablation Study
Tests different sizes to find the sweet spot.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_cnf import PointFlow2DCNF

def test_architecture_size(hidden_dim, solver_steps, data_path, slice_name, device='cuda'):
    """Test a specific architecture configuration."""
    
    # Load data
    slice_path = Path(data_path) / slice_name
    target_points = torch.from_numpy(np.load(slice_path)).float().to(device)
    if target_points.ndim == 1:
        target_points = target_points.reshape(-1, 2)
    
    # Normalize
    center = target_points.mean(dim=0)
    scale = (target_points - center).abs().max() * 1.1
    target_points = (target_points - center) / scale
    num_points = target_points.shape[0]
    
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
        solver_steps=solver_steps
    ).to(device)
    
    num_params = sum(p.numel() for p in decoder.parameters())
    
    # Train
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3)
    epochs = 150  # Quick test
    
    losses = []
    for epoch in range(epochs):
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
    
    return {
        'hidden_dim': hidden_dim,
        'solver_steps': solver_steps,
        'num_params': num_params,
        'final_loss': losses[-1],
        'best_loss': min(losses),
        'convergence_epoch': losses.index(min(losses)),
        'losses': losses
    }

def run_ablation_study(data_path: str, slice_name: str, device: str = 'cuda'):
    """Run comprehensive ablation study."""
    
    print("\n🧪 CNF ARCHITECTURE ABLATION STUDY")
    print("=" * 50)
    
    # Test configurations
    hidden_dims = [16, 32, 64, 128, 256]
    solver_steps = [5, 10, 20, 50]
    
    results = []
    
    print("\nTesting different architectures...")
    for hidden_dim in hidden_dims:
        for steps in solver_steps:
            print(f"\n📊 Testing: hidden_dim={hidden_dim}, solver_steps={steps}")
            
            try:
                result = test_architecture_size(hidden_dim, steps, data_path, slice_name, device)
                results.append(result)
                print(f"   Params: {result['num_params']:,}")
                print(f"   Best loss: {result['best_loss']:.4f} at epoch {result['convergence_epoch']}")
                print(f"   Final loss: {result['final_loss']:.4f}")
            except Exception as e:
                print(f"   Failed: {e}")
    
    # Save results
    output_dir = Path("outputs/cnf_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    # 1. Params vs Performance
    plt.subplot(2, 3, 1)
    params = [r['num_params'] for r in results]
    best_losses = [r['best_loss'] for r in results]
    plt.scatter(params, best_losses, s=100, alpha=0.6)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Best Loss')
    plt.title('Model Size vs Performance')
    plt.xscale('log')
    plt.grid(True)
    
    # 2. Hidden Dim Effect
    plt.subplot(2, 3, 2)
    for steps in solver_steps:
        step_results = [r for r in results if r['solver_steps'] == steps]
        if step_results:
            dims = [r['hidden_dim'] for r in step_results]
            losses = [r['best_loss'] for r in step_results]
            plt.plot(dims, losses, marker='o', label=f'{steps} steps')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Best Loss')
    plt.title('Hidden Dim Effect')
    plt.legend()
    plt.grid(True)
    
    # 3. Solver Steps Effect
    plt.subplot(2, 3, 3)
    for dim in hidden_dims:
        dim_results = [r for r in results if r['hidden_dim'] == dim]
        if dim_results:
            steps = [r['solver_steps'] for r in dim_results]
            losses = [r['best_loss'] for r in dim_results]
            plt.plot(steps, losses, marker='o', label=f'dim={dim}')
    plt.xlabel('Solver Steps')
    plt.ylabel('Best Loss')
    plt.title('Integration Steps Effect')
    plt.legend()
    plt.grid(True)
    
    # 4. Convergence Speed
    plt.subplot(2, 3, 4)
    convergence_epochs = [r['convergence_epoch'] for r in results]
    plt.scatter(params, convergence_epochs, s=100, alpha=0.6)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Convergence Epoch')
    plt.title('Model Size vs Convergence Speed')
    plt.xscale('log')
    plt.grid(True)
    
    # 5. Best configurations
    plt.subplot(2, 3, 5)
    sorted_results = sorted(results, key=lambda x: x['best_loss'])[:5]
    configs = [f"h={r['hidden_dim']},s={r['solver_steps']}" for r in sorted_results]
    losses = [r['best_loss'] for r in sorted_results]
    plt.bar(configs, losses)
    plt.xlabel('Configuration')
    plt.ylabel('Best Loss')
    plt.title('Top 5 Configurations')
    plt.xticks(rotation=45)
    
    # 6. Efficiency plot
    plt.subplot(2, 3, 6)
    efficiency = [r['best_loss'] * r['num_params'] / 1000 for r in results]
    plt.scatter(params, efficiency, s=100, alpha=0.6)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Loss × Params / 1000')
    plt.title('Efficiency Score (lower is better)')
    plt.xscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_results.png", dpi=150)
    plt.close()
    
    # Print summary
    print("\n📊 ABLATION STUDY SUMMARY")
    print("=" * 50)
    print("\nTop 5 configurations:")
    for i, r in enumerate(sorted_results[:5]):
        print(f"{i+1}. Hidden={r['hidden_dim']}, Steps={r['solver_steps']}")
        print(f"   Params: {r['num_params']:,}, Loss: {r['best_loss']:.4f}")
    
    # Find sweet spot
    sweet_spot = min(results, key=lambda x: x['best_loss'] * np.log(x['num_params']))
    print(f"\n🎯 Recommended configuration (best efficiency):")
    print(f"   Hidden dim: {sweet_spot['hidden_dim']}")
    print(f"   Solver steps: {sweet_spot['solver_steps']}")
    print(f"   Parameters: {sweet_spot['num_params']:,}")
    print(f"   Best loss: {sweet_spot['best_loss']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_ablation_study(args.data_path, args.slice_name, args.device)
