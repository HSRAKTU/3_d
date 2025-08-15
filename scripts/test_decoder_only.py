#!/usr/bin/env python3
"""
Test JUST the decoder - can it learn to decode a fixed latent to a slice?
This is the minimal test to prove our decoder works.
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

from models.pointflow_cnf import PointFlowCNF

def test_decoder_learning(data_path: str, slice_name: str, device: str = 'cuda'):
    """Test if decoder can learn to map a FIXED latent code to a slice."""
    
    print("\n🎯 DECODER-ONLY TEST")
    print("=" * 50)
    print("Goal: Prove the decoder can learn z → slice mapping")
    print("Method: Fix a random z, train decoder to output target slice")
    
    # Load target slice
    slice_path = Path(data_path) / slice_name
    
    # Handle different data formats
    if slice_path.suffix == '.npy':
        # NumPy format
        target_points = torch.from_numpy(np.load(slice_path)).float().to(device)
        if target_points.ndim == 1:
            # Flatten array, reshape to [N, 2]
            target_points = target_points.reshape(-1, 2)
        mask = torch.ones(target_points.shape[0]).to(device)  # All points are valid
    else:
        # PyTorch format
        data = torch.load(slice_path, map_location=device)
        if isinstance(data, dict):
            target_points = data['points'].float().to(device)  # [N, 2]
            mask = data.get('mask', torch.ones(target_points.shape[0])).float().to(device)  # [N]
        else:
            # Just a tensor
            target_points = data.float().to(device)
            if target_points.ndim == 1:
                target_points = target_points.reshape(-1, 2)
            mask = torch.ones(target_points.shape[0]).to(device)
    
    # Get actual points (remove padding)
    valid_points = target_points[mask.bool()]
    num_points = valid_points.shape[0]
    print(f"\n📊 Target slice: {num_points} points")
    
    # Normalize to [-1, 1]
    center = valid_points.mean(dim=0)
    scale = (valid_points - center).abs().max()
    valid_points = (valid_points - center) / scale
    
    # FIXED random latent code
    latent_dim = 128
    fixed_z = torch.randn(1, latent_dim).to(device)
    print(f"🔒 Fixed latent code: z ~ N(0,1), dim={latent_dim}")
    
    # Create decoder only
    decoder = PointFlowCNF(
        point_dim=2,
        context_dim=latent_dim,
        hidden_dim=256,  # Bigger than default
        solver='dopri5',
        atol=1e-3,  # Relaxed for stability
        rtol=1e-3
    ).to(device)
    
    print(f"🏗️  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    
    # Training loop
    epochs = 1000
    losses = []
    
    print("\n🚀 Training decoder to reconstruct fixed slice...")
    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Generate points from fixed latent
        generated = decoder.sample(fixed_z, num_points)  # [1, N, 2]
        generated = generated.squeeze(0)  # [N, 2]
        
        # Chamfer distance (symmetric nearest neighbor)
        # Forward: for each generated point, find nearest target
        dist_gen_to_target = torch.cdist(generated, valid_points).min(dim=1)[0]
        # Backward: for each target point, find nearest generated  
        dist_target_to_gen = torch.cdist(valid_points, generated).min(dim=1)[0]
        
        # Symmetric loss
        loss = dist_gen_to_target.mean() + dist_target_to_gen.mean()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # Visualize progress
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch}: Loss = {loss.item():.4f}")
            
            # Save visualization
            with torch.no_grad():
                generated_vis = decoder.sample(fixed_z, num_points).squeeze(0).cpu()
                
                plt.figure(figsize=(10, 5))
                
                # Original
                plt.subplot(1, 2, 1)
                plt.scatter(valid_points.cpu()[:, 0], valid_points.cpu()[:, 1], 
                           alpha=0.6, s=20, label='Target')
                plt.title(f'Target Slice ({num_points} points)')
                plt.axis('equal')
                plt.grid(True, alpha=0.3)
                plt.xlim(-1.5, 1.5)
                plt.ylim(-1.5, 1.5)
                
                # Generated
                plt.subplot(1, 2, 2)
                plt.scatter(generated_vis[:, 0], generated_vis[:, 1], 
                           alpha=0.6, s=20, color='red', label='Generated')
                plt.title(f'Epoch {epoch}: Loss={loss.item():.4f}')
                plt.axis('equal')
                plt.grid(True, alpha=0.3)
                plt.xlim(-1.5, 1.5)
                plt.ylim(-1.5, 1.5)
                
                plt.tight_layout()
                
                output_dir = Path("outputs/decoder_test")
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=100)
                plt.close()
    
    # Final evaluation
    final_loss = losses[-1]
    print(f"\n✅ Training complete!")
    print(f"📊 Final loss: {final_loss:.4f}")
    print(f"📉 Loss reduction: {losses[0]:.4f} → {final_loss:.4f} ({losses[0]/final_loss:.1f}x)")
    
    # Plot loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Chamfer Distance')
    plt.title('Decoder Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()
    
    if final_loss < 0.1:
        print("\n🎉 SUCCESS! Decoder learned to reconstruct the slice!")
    else:
        print("\n⚠️  Decoder struggling. May need more epochs or tuning.")
    
    return final_loss < 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to data directory')
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Test decoder
    success = test_decoder_learning(args.data_path, args.slice_name, args.device)
    
    if success:
        print("\n✅ Decoder test PASSED! Moving on to full VAE...")
    else:
        print("\n❌ Decoder test FAILED! Debug before proceeding.")
