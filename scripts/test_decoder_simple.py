#!/usr/bin/env python3
"""
SIMPLEST possible decoder test - start with just 50 points to prove it can learn.
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

def test_simple_decoder(data_path: str, slice_name: str, device: str = 'cuda'):
    """Test decoder on simplified problem first."""
    
    print("\n🎯 SIMPLE DECODER TEST - Start Small")
    print("=" * 50)
    
    # Load and subsample points
    slice_path = Path(data_path) / slice_name
    all_points = torch.from_numpy(np.load(slice_path)).float()
    if all_points.ndim == 1:
        all_points = all_points.reshape(-1, 2)
    
    # START WITH JUST 50 POINTS!
    indices = torch.randperm(all_points.shape[0])[:50]
    target_points = all_points[indices].to(device)
    
    print(f"📊 Using only {target_points.shape[0]} points (simplified problem)")
    
    # Normalize
    center = target_points.mean(dim=0)
    scale = (target_points - center).abs().max() * 1.2
    target_points = (target_points - center) / scale
    
    # Fixed latent
    latent_dim = 32  # Even smaller latent space
    fixed_z = torch.randn(1, latent_dim).to(device) * 0.5  # Smaller magnitude
    
    # VERY simple decoder
    decoder = PointFlowCNF(
        point_dim=2,
        context_dim=latent_dim,
        hidden_dim=64,  # Very simple
        solver='dopri5',
        atol=1e-2,  # Very relaxed
        rtol=1e-2
    ).to(device)
    
    print(f"🏗️  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # High learning rate for simple problem
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)
    
    # Quick training
    epochs = 500
    losses = []
    
    print("\n🚀 Training on simplified problem...")
    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Generate
        generated = decoder.sample(fixed_z, 50).squeeze(0)
        
        # Loss
        dist_g2t = torch.cdist(generated, target_points).min(dim=1)[0].mean()
        dist_t2g = torch.cdist(target_points, generated).min(dim=1)[0].mean()
        loss = dist_g2t + dist_t2g
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch}: Loss = {loss.item():.4f}")
    
    final_loss = losses[-1]
    print(f"\n📊 Final loss: {final_loss:.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.grid(True)
    
    # Final result
    with torch.no_grad():
        final_gen = decoder.sample(fixed_z, 50).squeeze(0).cpu()
    
    plt.subplot(1, 3, 2)
    plt.scatter(target_points.cpu()[:, 0], target_points.cpu()[:, 1], s=50, alpha=0.7, label='Target')
    plt.scatter(final_gen[:, 0], final_gen[:, 1], s=30, alpha=0.7, label='Generated')
    plt.legend()
    plt.axis('equal')
    plt.title(f'Final Result (Loss={final_loss:.3f})')
    plt.grid(True)
    
    # Distance histogram
    plt.subplot(1, 3, 3)
    distances = torch.cdist(final_gen, target_points.cpu()).min(dim=1)[0]
    plt.hist(distances.numpy(), bins=30)
    plt.xlabel('Distance to nearest target')
    plt.title(f'Distance Distribution (mean={distances.mean():.3f})')
    
    plt.tight_layout()
    output_dir = Path("outputs/decoder_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "results.png", dpi=150)
    plt.close()
    
    if final_loss < 0.1:
        print("\n✅ Simple test PASSED! Decoder CAN learn.")
        print("→ Now try the full 584 points with these settings")
        return True
    else:
        print("\n❌ Even simple test failed. Major architecture issues.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    test_simple_decoder(args.data_path, args.slice_name, args.device)
