#!/usr/bin/env python3
"""
Test Latent CNF behavior - the other CNF we haven't examined yet.
This transforms the latent distribution to match a prior.
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

from models.latent_cnf import LatentCNF
from models.encoder import PointNet2DEncoder

def test_latent_cnf(data_path: str, slice_name: str, device: str = 'cuda'):
    """Test how Latent CNF transforms encoder outputs."""
    
    print("\n🧪 LATENT CNF BEHAVIOR TEST")
    print("=" * 50)
    print("Testing the CNF that transforms latent codes to prior distribution")
    
    # Load data
    slice_path = Path(data_path) / slice_name
    points = torch.from_numpy(np.load(slice_path)).float().to(device)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    
    # Normalize
    center = points.mean(dim=0)
    scale = (points - center).abs().max() * 1.1
    points = (points - center) / scale
    
    # Create encoder
    latent_dim = 32
    encoder = PointNet2DEncoder(input_dim=2, latent_dim=latent_dim).to(device)
    
    # Test different Latent CNF configurations
    hidden_dims = [32, 64, 128]
    results = {}
    
    for hidden_dim in hidden_dims:
        print(f"\n📊 Testing Latent CNF with hidden_dim={hidden_dim}")
        
        # Create Latent CNF
        latent_cnf = LatentCNF(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            solver='dopri5',
            atol=1e-3,
            rtol=1e-3
        ).to(device)
        
        print(f"   Parameters: {sum(p.numel() for p in latent_cnf.parameters()):,}")
        
        # Encode some points
        with torch.no_grad():
            # Add batch dimension
            points_batch = points.unsqueeze(0)
            mu, logvar = encoder(points_batch)
            
            # Sample z
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            
            # Transform through Latent CNF
            w, delta_log_pw = latent_cnf(z, None, torch.zeros(1, 1).to(device))
        
        # Analyze distributions
        z_mean = z.mean().item()
        z_std = z.std().item()
        w_mean = w.mean().item()
        w_std = w.std().item()
        
        print(f"   z distribution: mean={z_mean:.3f}, std={z_std:.3f}")
        print(f"   w distribution: mean={w_mean:.3f}, std={w_std:.3f}")
        print(f"   Delta log p: {delta_log_pw.item():.3f}")
        
        # Test if w is closer to standard normal
        w_normality = abs(w_mean) + abs(w_std - 1.0)
        print(f"   Normality score: {w_normality:.3f} (lower is better)")
        
        results[hidden_dim] = {
            'params': sum(p.numel() for p in latent_cnf.parameters()),
            'z_stats': (z_mean, z_std),
            'w_stats': (w_mean, w_std),
            'delta_log_p': delta_log_pw.item(),
            'normality': w_normality
        }
    
    # Visualize
    output_dir = Path("outputs/latent_cnf_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test transformation with multiple samples
    print("\n📊 Testing with multiple slices...")
    
    # Load more slices if available
    data_dir = Path(data_path)
    slice_files = list(data_dir.glob("*.npy"))[:10]
    
    if len(slice_files) > 1:
        all_z = []
        all_w = []
        
        # Use best config
        best_dim = min(results.keys(), key=lambda k: results[k]['normality'])
        latent_cnf = LatentCNF(latent_dim=latent_dim, hidden_dim=best_dim).to(device)
        
        for slice_file in slice_files:
            points = torch.from_numpy(np.load(slice_file)).float().to(device)
            if points.ndim == 1:
                points = points.reshape(-1, 2)
            
            # Normalize
            points = (points - points.mean(dim=0)) / points.abs().max()
            
            with torch.no_grad():
                mu, logvar = encoder(points.unsqueeze(0))
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                w, _ = latent_cnf(z, None, torch.zeros(1, 1).to(device))
                
                all_z.append(z.cpu())
                all_w.append(w.cpu())
        
        all_z = torch.cat(all_z)
        all_w = torch.cat(all_w)
        
        # Visualize distributions
        plt.figure(figsize=(12, 5))
        
        # Z distribution
        plt.subplot(1, 2, 1)
        plt.hist2d(all_z[:, 0], all_z[:, 1], bins=30, cmap='Blues')
        plt.colorbar()
        plt.title('Z Distribution (Encoder Output)')
        plt.xlabel('Dim 0')
        plt.ylabel('Dim 1')
        
        # W distribution
        plt.subplot(1, 2, 2)
        plt.hist2d(all_w[:, 0], all_w[:, 1], bins=30, cmap='Reds')
        plt.colorbar()
        plt.title('W Distribution (After Latent CNF)')
        plt.xlabel('Dim 0')
        plt.ylabel('Dim 1')
        
        plt.tight_layout()
        plt.savefig(output_dir / "latent_distributions.png", dpi=150)
        plt.close()
    
    # Summary
    print("\n📊 LATENT CNF SUMMARY")
    print("=" * 50)
    for dim, res in results.items():
        print(f"\nHidden dim {dim}:")
        print(f"  Parameters: {res['params']:,}")
        print(f"  Normality score: {res['normality']:.3f}")
    
    best_config = min(results.items(), key=lambda x: x[1]['normality'])
    print(f"\n🎯 Best configuration: hidden_dim={best_config[0]}")
    print(f"   This transforms encoder outputs closest to standard normal")
    
    # Test if we even need Latent CNF
    print("\n🤔 Do we even need Latent CNF?")
    print("If encoder outputs are already near-normal, we might skip it!")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    test_latent_cnf(args.data_path, args.slice_name, args.device)
