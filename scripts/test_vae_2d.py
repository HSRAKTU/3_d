#!/usr/bin/env python3
"""
Quick test script for PointFlow2D VAE implementation.
Tests basic functionality before POD deployment.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our VAE implementation
from train_vae_2d import PointFlow2DVAE, compute_chamfer_distance


def test_vae_basic():
    """Test basic VAE functionality"""
    print("Testing PointFlow2D VAE...")
    
    # Create model
    model = PointFlow2DVAE(
        latent_dim=64,
        cnf_hidden_dim=128,
        solver_steps=5,
        use_stochastic=False  # Start deterministic
    )
    
    # Test data
    batch_size = 2
    num_points = 100
    x = torch.randn(batch_size, num_points, 2)
    
    print("\n1. Testing encode...")
    z, mu, logvar = model.encode(x)
    print(f"   Latent shape: {z.shape}")
    print(f"   Latent mean: {z.mean().item():.4f}")
    print(f"   Latent std: {z.std().item():.4f}")
    
    print("\n2. Testing CNF forward...")
    y, log_det = model.cnf_forward(x, z)
    print(f"   Output shape: {y.shape}")
    print(f"   Output mean: {y.mean().item():.4f}")
    print(f"   Output std: {y.std().item():.4f}")
    print(f"   Log det mean: {log_det.mean().item():.4f}")
    
    print("\n3. Testing sampling...")
    x_recon = model.sample(z, num_points)
    print(f"   Reconstructed shape: {x_recon.shape}")
    
    print("\n4. Testing losses...")
    loss, losses = model.compute_losses(x)
    print(f"   Total loss: {loss.item():.4f}")
    for k, v in losses.items():
        if k != 'total':
            print(f"   {k}: {v:.4f}")
    
    print("\n5. Testing Chamfer distance...")
    chamfer = compute_chamfer_distance(x_recon, x)
    print(f"   Chamfer distance: {chamfer.item():.4f}")
    
    # Test stochastic mode
    print("\n6. Testing stochastic encoder...")
    model_stochastic = PointFlow2DVAE(
        latent_dim=64,
        cnf_hidden_dim=128,
        solver_steps=5,
        use_stochastic=True
    )
    
    z, mu, logvar = model_stochastic.encode(x)
    if mu is not None:
        print(f"   Mu shape: {mu.shape}")
        print(f"   Logvar shape: {logvar.shape}")
        print(f"   KL divergence: {0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(dim=1).mean().item():.4f}")
    else:
        print("   Using deterministic encoder")
    
    print("\n✅ All tests passed!")
    

def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model = PointFlow2DVAE(
        latent_dim=128,
        solver_steps=10,
        use_stochastic=False
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test batch
    x = torch.randn(4, 200, 2).to(device)
    
    # Forward pass
    loss, losses = model.compute_losses(x)
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"Gradient norm: {total_grad_norm:.4f}")
    
    # Update
    optimizer.step()
    
    # Check loss after update
    # Note: We need gradients for divergence computation
    loss_after, _ = model.compute_losses(x)
    print(f"Loss after update: {loss_after.item():.4f}")
    print(f"Loss decreased: {loss.item() > loss_after.item()}")
    
    print("\n✅ Training step test passed!")


def test_memory_efficiency():
    """Test memory usage with different batch sizes"""
    print("\nTesting memory efficiency...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
        
    device = torch.device('cuda')
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = PointFlow2DVAE(
        latent_dim=128,
        solver_steps=10
    ).to(device)
    
    batch_sizes = [4, 8, 16, 32]
    num_points = 500
    
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            x = torch.randn(bs, num_points, 2).to(device)
            loss, _ = model.compute_losses(x)
            loss.backward()
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"Batch size {bs}: Peak memory {peak_memory:.1f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {bs}: OOM")
                break
            else:
                raise e
    
    print("\n✅ Memory test complete!")


if __name__ == '__main__':
    print("="*50)
    print("PointFlow2D VAE Test Suite")
    print("="*50)
    
    test_vae_basic()
    test_training_step()
    test_memory_efficiency()
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("Ready for POD deployment!")
    print("="*50)
