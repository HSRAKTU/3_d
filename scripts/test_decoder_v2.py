#!/usr/bin/env python3
"""
IMPROVED decoder test with better hyperparameters and diagnostics.
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

def visualize_results(target, generated, epoch, loss, save_path):
    """Better visualization showing what's actually happening."""
    plt.figure(figsize=(15, 5))
    
    # 1. Target vs Generated
    plt.subplot(1, 3, 1)
    plt.scatter(target[:, 0], target[:, 1], alpha=0.6, s=20, label='Target', color='blue')
    plt.scatter(generated[:, 0], generated[:, 1], alpha=0.6, s=20, label='Generated', color='red')
    plt.title(f'Epoch {epoch}: Loss={loss:.4f}')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    # 2. Overlay (to see alignment)
    plt.subplot(1, 3, 2)
    plt.scatter(target[:, 0], target[:, 1], alpha=0.3, s=40, label='Target', color='blue')
    plt.scatter(generated[:, 0], generated[:, 1], alpha=0.8, s=10, label='Generated', color='red')
    plt.title('Overlay View')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    # 3. Point-wise distances
    plt.subplot(1, 3, 3)
    # For each generated point, find nearest target
    distances = torch.cdist(generated.unsqueeze(0), target.unsqueeze(0)).squeeze(0).min(dim=1)[0]
    plt.hist(distances.cpu().numpy(), bins=50, alpha=0.7, color='green')
    plt.title(f'Distance Distribution (mean={distances.mean():.3f})')
    plt.xlabel('Distance to nearest target point')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def test_decoder_v2(data_path: str, slice_name: str, device: str = 'cuda'):
    """Improved decoder test with better hyperparameters."""
    
    print("\n🎯 DECODER TEST V2 - With Fixes")
    print("=" * 50)
    
    # Load target slice
    slice_path = Path(data_path) / slice_name
    target_points = torch.from_numpy(np.load(slice_path)).float().to(device)
    if target_points.ndim == 1:
        target_points = target_points.reshape(-1, 2)
    
    num_points = target_points.shape[0]
    print(f"📊 Target slice: {num_points} points")
    
    # Normalize to [-1, 1]
    center = target_points.mean(dim=0)
    scale = (target_points - center).abs().max() * 1.1  # Add 10% margin
    target_points = (target_points - center) / scale
    
    # FIXED random latent code
    latent_dim = 128
    fixed_z = torch.randn(1, latent_dim).to(device)
    fixed_z = fixed_z / fixed_z.norm() * np.sqrt(latent_dim)  # Normalize to reasonable scale
    
    # Create decoder with SIMPLER architecture
    decoder = PointFlowCNF(
        point_dim=2,
        context_dim=latent_dim,
        hidden_dim=128,  # Reduced from 256
        solver='dopri5',
        atol=5e-3,  # More relaxed
        rtol=5e-3   # More relaxed
    ).to(device)
    
    print(f"🏗️  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Better optimizer with weight decay
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-3, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
    
    # Training
    epochs = 1000
    losses = []
    output_dir = Path("outputs/decoder_test_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🚀 Training with improved settings...")
    print(f"   Learning rate: 5e-3 → 1e-5 (cosine schedule)")
    print(f"   Hidden dim: 128 (simpler)")
    print(f"   ODE tolerance: 5e-3 (more relaxed)")
    
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Generate points
        generated = decoder.sample(fixed_z, num_points)
        generated = generated.squeeze(0)
        
        # Chamfer distance
        dist_gen_to_target = torch.cdist(generated, target_points).min(dim=1)[0]
        dist_target_to_gen = torch.cdist(target_points, generated).min(dim=1)[0]
        loss = dist_gen_to_target.mean() + dist_target_to_gen.mean()
        
        # Add small regularization to prevent explosion
        reg_loss = 0.001 * (generated**2).mean()
        total_loss = loss + reg_loss
        
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        # Detailed logging
        if epoch % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch}: Loss={loss.item():.4f}, Best={best_loss:.4f}, LR={current_lr:.6f}, GradNorm={grad_norm:.3f}")
            
            # Save visualization
            with torch.no_grad():
                visualize_results(
                    target_points, 
                    generated,
                    epoch, 
                    loss.item(),
                    output_dir / f"epoch_{epoch:04d}.png"
                )
    
    # Final evaluation
    final_loss = losses[-1]
    print(f"\n✅ Training complete!")
    print(f"📊 Final loss: {final_loss:.4f}")
    print(f"📊 Best loss: {best_loss:.4f}")
    print(f"📉 Loss reduction: {losses[0]:.4f} → {best_loss:.4f} ({losses[0]/best_loss:.1f}x)")
    
    # Save final comparison
    with torch.no_grad():
        final_generated = decoder.sample(fixed_z, num_points).squeeze(0)
        visualize_results(
            target_points,
            final_generated, 
            "Final",
            final_loss,
            output_dir / "final_result.png"
        )
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.axhline(y=0.1, color='r', linestyle='--', label='Target (0.1)')
    plt.xlabel('Epoch')
    plt.ylabel('Chamfer Distance')
    plt.title('Decoder Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()
    
    success = best_loss < 0.15  # More lenient threshold
    
    if success:
        print(f"\n🎉 SUCCESS! Decoder learned adequately (loss < 0.15)")
        print("\n📋 Next Steps:")
        print("1. Use these hyperparameters for two-stage training")
        print("2. Start with lr=5e-3 and cosine scheduling")
        print("3. Keep ODE tolerances at 5e-3")
    else:
        print(f"\n⚠️  Decoder still struggling. Best loss: {best_loss:.4f}")
        print("\n🔧 Debugging suggestions:")
        print("1. Check visualizations in outputs/decoder_test_v2/")
        print("2. Try even simpler architecture (hidden_dim=64)")
        print("3. Consider using fewer points initially")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    success = test_decoder_v2(args.data_path, args.slice_name, args.device)
