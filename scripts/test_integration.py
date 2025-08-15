#!/usr/bin/env python3
"""
Integration Test: Encoder + Decoder together (without full VAE complexity).
Tests if the components work well together.
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

from models.encoder import PointNet2DEncoder
from models.pointflow2d_cnf import PointFlow2DCNF

def test_encoder_decoder_integration(data_path: str, slice_name: str, device: str = 'cuda'):
    """Test encoder + decoder working together."""
    
    print("\n🧪 ENCODER + DECODER INTEGRATION TEST")
    print("=" * 50)
    
    # Load data
    slice_path = Path(data_path) / slice_name
    target_points = torch.from_numpy(np.load(slice_path)).float().to(device)
    if target_points.ndim == 1:
        target_points = target_points.reshape(-1, 2)
    
    num_points = target_points.shape[0]
    
    # Normalize
    center = target_points.mean(dim=0)
    scale = (target_points - center).abs().max() * 1.1
    target_points = (target_points - center) / scale
    
    # Test different latent dimensions
    latent_dims = [8, 16, 32, 64]
    results = {}
    
    for latent_dim in latent_dims:
        print(f"\n📊 Testing with latent_dim={latent_dim}")
        
        # Create encoder and decoder
        encoder = PointNet2DEncoder(
            input_dim=2,
            latent_dim=latent_dim,
            hidden_dim=128
        ).to(device)
        
        decoder = PointFlow2DCNF(
            point_dim=2,
            context_dim=latent_dim,
            hidden_dim=64,
            solver='euler',
            solver_steps=20
        ).to(device)
        
        total_params = sum(p.numel() for p in encoder.parameters()) + \
                      sum(p.numel() for p in decoder.parameters())
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   (Encoder: {sum(p.numel() for p in encoder.parameters()):,}, " +
              f"Decoder: {sum(p.numel() for p in decoder.parameters()):,})")
        
        # Joint optimizer
        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        
        # Training
        epochs = 200
        losses = []
        latent_norms = []
        
        for epoch in tqdm(range(epochs), desc=f"Latent {latent_dim}"):
            optimizer.zero_grad()
            
            # Encode
            points_batch = target_points.unsqueeze(0)
            mu, logvar = encoder(points_batch)
            
            # Use deterministic encoding for now
            z = mu
            
            # Decode
            reconstructed = decoder.sample(z, num_points).squeeze(0)
            
            # Chamfer loss
            dist_g2t = torch.cdist(reconstructed, target_points).min(dim=1)[0].mean()
            dist_t2g = torch.cdist(target_points, reconstructed).min(dim=1)[0].mean()
            loss = dist_g2t + dist_t2g
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            
            losses.append(loss.item())
            latent_norms.append(z.norm().item())
        
        # Analyze final reconstruction
        with torch.no_grad():
            mu_final, _ = encoder(points_batch)
            recon_final = decoder.sample(mu_final, num_points).squeeze(0)
            
            # Compute final metrics
            final_chamfer = (torch.cdist(recon_final, target_points).min(dim=1)[0].mean() + 
                           torch.cdist(target_points, recon_final).min(dim=1)[0].mean()).item()
        
        results[latent_dim] = {
            'total_params': total_params,
            'final_loss': losses[-1],
            'best_loss': min(losses),
            'convergence_epoch': losses.index(min(losses)),
            'final_chamfer': final_chamfer,
            'latent_norm_mean': np.mean(latent_norms),
            'latent_norm_std': np.std(latent_norms),
            'losses': losses
        }
        
        print(f"   Final loss: {losses[-1]:.4f}")
        print(f"   Best loss: {min(losses):.4f} at epoch {losses.index(min(losses))}")
        print(f"   Latent norm: {np.mean(latent_norms):.2f} ± {np.std(latent_norms):.2f}")
    
    # Visualize results
    output_dir = Path("outputs/integration_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Loss curves
    plt.subplot(2, 3, 1)
    for latent_dim, res in results.items():
        plt.plot(res['losses'], label=f'Latent {latent_dim}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 2. Latent dim vs performance
    plt.subplot(2, 3, 2)
    dims = list(results.keys())
    final_losses = [results[d]['final_loss'] for d in dims]
    best_losses = [results[d]['best_loss'] for d in dims]
    plt.plot(dims, final_losses, 'o-', label='Final', markersize=10)
    plt.plot(dims, best_losses, 's-', label='Best', markersize=10)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Loss')
    plt.title('Latent Size Effect')
    plt.legend()
    plt.grid(True)
    
    # 3. Convergence speed
    plt.subplot(2, 3, 3)
    conv_epochs = [results[d]['convergence_epoch'] for d in dims]
    plt.bar([str(d) for d in dims], conv_epochs)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Convergence Epoch')
    plt.title('Convergence Speed')
    
    # 4. Latent norm behavior
    plt.subplot(2, 3, 4)
    norm_means = [results[d]['latent_norm_mean'] for d in dims]
    norm_stds = [results[d]['latent_norm_std'] for d in dims]
    plt.errorbar(dims, norm_means, yerr=norm_stds, fmt='o-', capsize=5)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Norm')
    plt.title('Latent Magnitude')
    plt.grid(True)
    
    # 5. Efficiency
    plt.subplot(2, 3, 5)
    params = [results[d]['total_params'] for d in dims]
    efficiency = [results[d]['best_loss'] * results[d]['total_params'] / 10000 for d in dims]
    plt.scatter(params, efficiency, s=100)
    for i, d in enumerate(dims):
        plt.annotate(f'L={d}', (params[i], efficiency[i]))
    plt.xlabel('Total Parameters')
    plt.ylabel('Loss × Params / 10k')
    plt.title('Efficiency (lower is better)')
    plt.grid(True)
    
    # 6. Final reconstructions
    plt.subplot(2, 3, 6)
    best_dim = min(results.keys(), key=lambda k: results[k]['best_loss'])
    
    # Re-run best config for visualization
    encoder = PointNet2DEncoder(input_dim=2, latent_dim=best_dim, hidden_dim=128).to(device)
    decoder = PointFlow2DCNF(point_dim=2, context_dim=best_dim, hidden_dim=64, 
                            solver='euler', solver_steps=20).to(device)
    
    # Quick train
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    for _ in range(100):
        optimizer.zero_grad()
        mu, _ = encoder(target_points.unsqueeze(0))
        recon = decoder.sample(mu, num_points).squeeze(0)
        loss = torch.cdist(recon, target_points).min(dim=1)[0].mean() + \
               torch.cdist(target_points, recon).min(dim=1)[0].mean()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        mu, _ = encoder(target_points.unsqueeze(0))
        final_recon = decoder.sample(mu, num_points).squeeze(0).cpu()
    
    target_cpu = target_points.cpu()
    plt.scatter(target_cpu[:, 0], target_cpu[:, 1], s=20, alpha=0.5, label='Target')
    plt.scatter(final_recon[:, 0], final_recon[:, 1], s=20, alpha=0.5, label='Reconstructed')
    plt.legend()
    plt.axis('equal')
    plt.title(f'Best Config (Latent={best_dim})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "integration_results.png", dpi=150)
    plt.close()
    
    # Summary
    print("\n📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    for dim in sorted(results.keys()):
        res = results[dim]
        print(f"\nLatent dimension {dim}:")
        print(f"  Best loss: {res['best_loss']:.4f}")
        print(f"  Parameters: {res['total_params']:,}")
        print(f"  Convergence: epoch {res['convergence_epoch']}")
    
    print(f"\n🎯 Best configuration: latent_dim={best_dim}")
    print(f"   This gives best reconstruction with reasonable parameter count")
    
    # Test information bottleneck
    print("\n💡 Information Bottleneck Analysis:")
    print(f"Input information: {num_points} points × 2 coords = {num_points * 2} values")
    for dim in results.keys():
        compression = (num_points * 2) / dim
        print(f"Latent {dim}: compression ratio = {compression:.1f}:1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    test_encoder_decoder_integration(args.data_path, args.slice_name, args.device)
