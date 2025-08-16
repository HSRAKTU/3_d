#!/usr/bin/env python3
"""
Single slice overfitting with PointFlow2D VAE.
Quick validation that the VAE approach works.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_vae_2d import PointFlow2DVAE, compute_chamfer_distance

# Configuration
LATENT_DIM = 128
HIDDEN_DIM = 256
SOLVER_STEPS = 10
LEARNING_RATE = 1e-3
EPOCHS = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_single_slice():
    """Load the standard test slice"""
    data_path = Path('data/scaled_slices/car_0/slice_z_025.npy')
    
    if data_path.exists():
        points = np.load(data_path)
        if isinstance(points, dict):
            points = points['points']
        return torch.tensor(points, dtype=torch.float32)
    else:
        # Generate random slice for testing
        print("Warning: Using random data")
        return torch.randn(584, 2) * 0.5

def train_single_slice():
    """Train on single slice"""
    print("Loading data...")
    target = load_single_slice().unsqueeze(0).to(DEVICE)  # [1, N, 2]
    print(f"Slice shape: {target.shape}")
    
    # Create model
    print("\nCreating model...")
    model = PointFlow2DVAE(
        latent_dim=LATENT_DIM,
        cnf_hidden_dim=HIDDEN_DIM,
        solver_steps=SOLVER_STEPS,
        use_stochastic=False,  # Start deterministic
        lambda_recon=1.0,
        lambda_prior=0.01,     # Small for single slice
        lambda_chamfer=50.0,   # High for overfitting
        lambda_volume=0.01
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    print("\nTraining...")
    best_chamfer = float('inf')
    history = {'loss': [], 'chamfer': [], 'recon': [], 'log_det': []}
    
    for epoch in range(EPOCHS):
        # Forward pass
        loss, losses = model.compute_losses(target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        with torch.no_grad():
            z, _, _ = model.encode(target)
            recon = model.sample(z, target.shape[1])
            chamfer = compute_chamfer_distance(recon, target).item()
            
        history['loss'].append(loss.item())
        history['chamfer'].append(chamfer)
        history['recon'].append(losses['recon'])
        history['log_det'].append(losses['log_det_mean'])
        
        # Update best
        if chamfer < best_chamfer:
            best_chamfer = chamfer
            best_epoch = epoch
            
        # Print progress
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():8.4f} | "
                  f"Chamfer: {chamfer:6.4f} | Recon: {losses['recon']:6.3f} | "
                  f"LogDet: {losses['log_det_mean']:6.3f}")
    
    print(f"\nBest Chamfer: {best_chamfer:.6f} at epoch {best_epoch}")
    
    # Final visualization
    visualize_results(model, target, history)
    
    return model, history

def visualize_results(model, target, history):
    """Visualize training results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].plot(history['chamfer'])
    axes[0, 1].set_title('Chamfer Distance')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].axhline(y=0.001, color='r', linestyle='--', label='Target')
    axes[0, 1].legend()
    
    axes[0, 2].plot(history['log_det'])
    axes[0, 2].set_title('Log Determinant')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].axhline(y=0, color='k', linestyle='--')
    
    # Point clouds
    with torch.no_grad():
        z, _, _ = model.encode(target)
        recon = model.sample(z, target.shape[1])
        
        # Also test generation from random z
        z_random = torch.randn_like(z)
        generated = model.sample(z_random, target.shape[1])
    
    target_np = target[0].cpu().numpy()
    recon_np = recon[0].cpu().numpy()
    generated_np = generated[0].cpu().numpy()
    
    # Original
    axes[1, 0].scatter(target_np[:, 0], target_np[:, 1], s=1, c='blue')
    axes[1, 0].set_title('Original Slice')
    axes[1, 0].set_aspect('equal')
    
    # Reconstruction
    axes[1, 1].scatter(recon_np[:, 0], recon_np[:, 1], s=1, c='red')
    axes[1, 1].set_title(f'Reconstruction (Chamfer: {history["chamfer"][-1]:.6f})')
    axes[1, 1].set_aspect('equal')
    
    # Generation
    axes[1, 2].scatter(generated_np[:, 0], generated_np[:, 1], s=1, c='green')
    axes[1, 2].set_title('Random Generation')
    axes[1, 2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('outputs/single_slice_vae_results.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    print("="*50)
    print("Single Slice VAE Overfitting Test")
    print("="*50)
    
    model, history = train_single_slice()
    
    print("\nTest complete!")
    print("Results saved to outputs/single_slice_vae_results.png")
