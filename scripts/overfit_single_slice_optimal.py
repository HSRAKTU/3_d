#!/usr/bin/env python3
"""
Single Slice Overfitting with OPTIMAL Configuration
Based on comprehensive test results
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from src.models.pointflow2d_cnf import PointFlow2DCNF
from src.models.encoder import PointNet2DEncoder

# Clear configuration based on test results
LATENT_DIM = 128  # As you intuited - enough capacity for 584 points
HIDDEN_DIM = 256  # Best performer in ablation study  
SOLVER_STEPS = 5  # Minimal steps with Euler
LEARNING_RATE = 1e-3
EPOCHS = 1000
TARGET_LOSS = 0.05

def load_single_slice():
    """Load a single slice for overfitting test"""
    data_path = Path("data/processed/slices/car_0/")
    slice_files = list(data_path.glob("slice_*.npy"))
    
    if not slice_files:
        raise FileNotFoundError(f"No slice files found in {data_path}")
    
    # Load first slice
    slice_data = np.load(slice_files[0], allow_pickle=True).item()
    points = torch.FloatTensor(slice_data['points'])  # [N, 2]
    
    # Normalize
    center = points.mean(dim=0)
    scale = (points - center).abs().max() * 1.1
    points = (points - center) / scale
    
    return points, center, scale

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load data
    print("\n📂 Loading single slice...")
    target_points, center, scale = load_single_slice()
    target_points = target_points.to(device)
    num_points = target_points.shape[0]
    print(f"✓ Loaded slice with {num_points} points")
    
    # Create models
    print("\n🏗️  Building optimal configuration...")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Solver: Euler with {SOLVER_STEPS} steps")
    
    # Encoder
    encoder = PointNet2DEncoder(
        point_dim=2,
        hidden_dim=128,
        latent_dim=LATENT_DIM
    ).to(device)
    
    # Decoder (the optimal lightweight CNF)
    decoder = PointFlow2DCNF(
        point_dim=2,
        context_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        solver='euler',
        solver_steps=SOLVER_STEPS
    ).to(device)
    
    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in decoder.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training
    print(f"\n🚀 Starting overfitting test (target loss: {TARGET_LOSS})")
    losses = []
    best_loss = float('inf')
    
    pbar = tqdm(range(EPOCHS), desc="Overfitting")
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Encode
        z_mu, z_logvar = encoder(target_points.unsqueeze(0))
        z = z_mu  # Deterministic for overfitting
        
        # Decode
        reconstructed = decoder.sample(z, num_points).squeeze(0)
        
        # Chamfer distance loss
        dist1 = torch.cdist(reconstructed, target_points).min(dim=1)[0].mean()
        dist2 = torch.cdist(target_points, reconstructed).min(dim=1)[0].mean()
        loss = (dist1 + dist2) / 2
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        # Track
        loss_val = loss.item()
        losses.append(loss_val)
        best_loss = min(best_loss, loss_val)
        
        pbar.set_postfix({
            'loss': f'{loss_val:.4f}',
            'best': f'{best_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.1e}'
        })
        
        # Early stopping if target reached
        if loss_val < TARGET_LOSS:
            print(f"\n🎯 Target loss reached at epoch {epoch}!")
            break
        
        # Visualize periodically
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            plt.figure(figsize=(15, 5))
            
            # Loss curve
            plt.subplot(1, 3, 1)
            plt.plot(losses)
            plt.axhline(y=TARGET_LOSS, color='r', linestyle='--', label=f'Target: {TARGET_LOSS}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            
            # Reconstruction
            with torch.no_grad():
                z_mu, _ = encoder(target_points.unsqueeze(0))
                recon = decoder.sample(z_mu, num_points).squeeze(0)
                
            plt.subplot(1, 3, 2)
            target_np = target_points.cpu().numpy()
            recon_np = recon.cpu().numpy()
            plt.scatter(target_np[:, 0], target_np[:, 1], c='blue', s=20, alpha=0.5, label='Target')
            plt.scatter(recon_np[:, 0], recon_np[:, 1], c='red', s=20, alpha=0.5, label='Reconstructed')
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.title(f'Epoch {epoch} (Loss: {loss_val:.4f})')
            
            # Overlay
            plt.subplot(1, 3, 3)
            plt.scatter(target_np[:, 0], target_np[:, 1], c='blue', s=40, alpha=0.3)
            plt.scatter(recon_np[:, 0], recon_np[:, 1], c='red', s=40, alpha=0.3)
            plt.axis('equal')
            plt.grid(True)
            plt.title('Overlay View')
            
            plt.tight_layout()
            
            # Save
            output_dir = Path("outputs/optimal_overfit")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    # Final evaluation
    print(f"\n📊 Final Results:")
    print(f"  Best loss achieved: {best_loss:.4f}")
    print(f"  Target loss: {TARGET_LOSS}")
    print(f"  Success: {'✓' if best_loss < TARGET_LOSS else '✗'}")
    
    # Save model if successful
    if best_loss < TARGET_LOSS:
        checkpoint = {
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict(),
            'config': {
                'latent_dim': LATENT_DIM,
                'hidden_dim': HIDDEN_DIM,
                'solver_steps': SOLVER_STEPS,
                'num_points': num_points,
                'best_loss': best_loss
            }
        }
        torch.save(checkpoint, output_dir / 'optimal_overfit_checkpoint.pth')
        print(f"\n💾 Model saved to {output_dir}/optimal_overfit_checkpoint.pth")
    
    # Plot final loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.axhline(y=TARGET_LOSS, color='r', linestyle='--', label=f'Target: {TARGET_LOSS}')
    plt.axhline(y=best_loss, color='g', linestyle='--', label=f'Best: {best_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Single Slice Overfitting - Optimal Configuration')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Overfitting test complete!")
    print(f"📁 Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
