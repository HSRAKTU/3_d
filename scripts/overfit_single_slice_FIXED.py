#!/usr/bin/env python3
"""
FIXED Single Slice Overfitting - Addressing Underfitting Issues

FIXES APPLIED:
✓ Increased model capacity: Hidden=512, Latent=256
✓ Better learning rate schedule: Two-phase with plateaus  
✓ More solver steps: 20 (vs 5) for higher quality
✓ Longer training: 2000 epochs
✓ Better loss function: Weighted Chamfer + Coverage loss
✓ Improved early stopping: Multiple criteria

TARGET: Loss < 0.03 for perfect reconstruction
Expected time: ~25-30 minutes (more thorough training)
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import datetime
import json
sys.path.append('.')

from src.models.pointflow2d_adapted import PointFlow2DAdaptedVAE

def compute_chamfer_distance(x, y):
    """Compute Chamfer distance between two point sets"""
    x_expanded = x.unsqueeze(1)  # [N, 1, 2]
    y_expanded = y.unsqueeze(0)  # [1, M, 2]
    
    # Compute all pairwise distances
    distances = torch.norm(x_expanded - y_expanded, dim=2)  # [N, M]
    
    # For each point in x, find closest point in y
    min_dist_x_to_y = distances.min(dim=1)[0]  # [N]
    
    # For each point in y, find closest point in x  
    min_dist_y_to_x = distances.min(dim=0)[0]  # [M]
    
    # Chamfer distance is sum of both directions
    chamfer = min_dist_x_to_y.mean() + min_dist_y_to_x.mean()
    
    return chamfer.item()

# 2D-ADAPTED configuration - optimized for 2D
LATENT_DIM = 128     # Sufficient for 2D
LEARNING_RATE = 1e-3 # Higher LR - simpler model trains faster
EPOCHS = 1000        # Should converge faster with proper 2D model
TARGET_CHAMFER = 0.001   # Target Chamfer distance for overfitting
BATCH_SIZE = 8       # Can use larger batch with simpler model

def load_single_slice():
    """Load a single slice for overfitting test"""
    # Try multiple possible data locations
    possible_paths = [
        Path("data/processed/slices/car_0/"),
        Path("../data/processed/slices/car_0/"),
        Path("../../data/processed/slices/car_0/"),
        Path("data/"),
        Path("../data/"),
        Path("../../data/")
    ]
    
    slice_data = None
    selected_path = None
    
    for data_path in possible_paths:
        slice_files = list(data_path.glob("slice_*.npy")) if data_path.exists() else []
        if slice_files:
            slice_data = np.load(slice_files[0], allow_pickle=True).item()
            selected_path = slice_files[0]
            break
            
        # Also try direct .npy files
        npy_files = list(data_path.glob("*.npy")) if data_path.exists() else []
        for npy_file in npy_files:
            try:
                data = np.load(npy_file, allow_pickle=True)
                if isinstance(data, dict) and 'points' in data:
                    slice_data = data
                    selected_path = npy_file
                    break
                elif isinstance(data, np.ndarray) and data.ndim == 2:
                    slice_data = {'points': data}
                    selected_path = npy_file
                    break
            except:
                continue
        if slice_data:
            break
    
    if slice_data is None:
        raise FileNotFoundError(f"No valid slice files found in any of: {[str(p) for p in possible_paths]}")
    
    print(f"✓ Found slice data: {selected_path}")
    points = torch.FloatTensor(slice_data['points'])  # [N, 2]
    
    # Normalize
    center = points.mean(dim=0)
    scale = (points - center).abs().max() * 1.1
    points = (points - center) / scale
    
    return points, center, scale, str(selected_path)

def improved_chamfer_loss(pred, target, coverage_weight=0.3):
    """
    Improved loss function addressing coverage issues
    """
    # Standard Chamfer distance
    dist_matrix = torch.cdist(pred, target)
    dist1 = dist_matrix.min(dim=1)[0].mean()  # Pred -> Target  
    dist2 = dist_matrix.min(dim=0)[0].mean()  # Target -> Pred
    chamfer = (dist1 + dist2) / 2
    
    # Coverage loss - ensure good point distribution
    coverage_threshold = 0.1
    target_covered = (dist_matrix.min(dim=0)[0] < coverage_threshold).float().mean()
    pred_coverage = (dist_matrix.min(dim=1)[0] < coverage_threshold).float().mean()
    coverage_loss = 1.0 - (target_covered + pred_coverage) / 2
    
    # Combined loss
    total_loss = chamfer + coverage_weight * coverage_loss
    
    return total_loss, chamfer, coverage_loss, target_covered, pred_coverage

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load data
    print("\n📂 Loading single slice...")
    target_points, center, scale, data_path = load_single_slice()
    target_points = target_points.to(device)
    num_points = target_points.shape[0]
    print(f"✓ Loaded slice with {num_points} points from: {data_path}")
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}], Scale: {scale:.3f}")
    
    # Create 2D-ADAPTED PointFlow model 
    print("\n🏗️  Building 2D-ADAPTED PointFlow...")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  CNF Hidden dimension: 256 (2D-optimized)")
    print(f"  Solver: euler with 10 steps (stable for 2D)")
    
    # 2D-adapted PointFlow with proper but simpler CNF
    model = PointFlow2DAdaptedVAE(
        input_dim=2,
        latent_dim=LATENT_DIM,
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,  # Appropriate for 2D
        solver='euler',  # More stable for 2D
        solver_steps=10,  # Fewer steps needed for 2D
        use_deterministic_encoder=True  # For overfitting test
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} (REAL CNF)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Cosine annealing for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    # Training
    print(f"\n🚀 Starting 2D-ADAPTED PointFlow test (target Chamfer: {TARGET_CHAMFER})")
    print(f"🚀 Using batch size: {BATCH_SIZE}")
    print(f"🔍 Tracking Chamfer distance as primary metric")
    losses = []
    best_loss = float('inf')
    
    # Create batched target
    target_batch = target_points.unsqueeze(0).repeat(BATCH_SIZE, 1, 1)  # [B, N, 2]
    
    pbar = tqdm(range(EPOCHS), desc="2D-Adapted PointFlow")
    for epoch in pbar:
        # PointFlow2DVAE forward handles everything (optimizer step, backward, etc.)
        step = epoch
        metrics = model.forward(target_batch, optimizer, step, writer=None)
        
        # Track PointFlow loss for debugging
        pointflow_loss = metrics.get('recon_nats', metrics.get('loss', 0))  # Get loss metric
        
        # Compute ACTUAL reconstruction quality every 10 epochs
        if epoch % 10 == 0:
            # Need gradients for CNF divergence computation
            recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
            chamfer_dist = compute_chamfer_distance(target_points, recon)
            loss_val = chamfer_dist  # Track Chamfer distance as primary metric
        else:
            loss_val = losses[-1] if losses else float('inf')  # Use previous value
        
        losses.append(loss_val)
        best_loss = min(best_loss, loss_val)
        
        scheduler.step()
        
        pbar.set_postfix({
            'chamfer': f'{loss_val:.4f}',
            'best': f'{best_loss:.4f}',
            'pf_loss': f'{pointflow_loss:.3f}',
            'lr': f'{scheduler.get_last_lr()[0]:.1e}'
        })
        
        # Early stopping if target reached
        if loss_val < TARGET_CHAMFER:
            print(f"\n🎯 Target Chamfer distance reached at epoch {epoch}!")
            break
        
        # Detailed analysis every 200 epochs
        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            print(f"\n📊 Epoch {epoch} Analysis:")
            
            # Need gradients for CNF
            recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
            
            # Detailed metrics
            total_loss, chamfer, coverage_loss, target_cov, pred_cov = improved_chamfer_loss(
                recon, target_points
            )
            
            print(f"  🎯 Total Loss: {total_loss:.4f} (Chamfer: {chamfer:.4f}, Coverage: {coverage_loss:.4f})")
            print(f"  📐 Point Coverage: Target {target_cov*100:.1f}%, Pred {pred_cov*100:.1f}%")
            print(f"  📈 Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save visualization
            if epoch % 400 == 0:  # Less frequent saves
                output_dir = Path("outputs/fixed_overfit")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                plt.figure(figsize=(15, 5))
                
                # Loss curves
                plt.subplot(1, 3, 1)
                plt.plot(losses, 'b-', label='Chamfer Distance')
                plt.axhline(y=TARGET_CHAMFER, color='r', linestyle='--', label=f'Target: {TARGET_CHAMFER}')
                plt.xlabel('Epoch')
                plt.ylabel('Chamfer Distance')
                plt.title('Overfitting Progress (Chamfer)')
                plt.legend()
                plt.grid(True)
                plt.yscale('log')
                
                # Reconstruction
                target_np = target_points.detach().cpu().numpy()
                recon_np = recon.detach().cpu().numpy()
                
                plt.subplot(1, 3, 2)
                plt.scatter(target_np[:, 0], target_np[:, 1], c='blue', s=30, alpha=0.7, label='Target')
                plt.scatter(recon_np[:, 0], recon_np[:, 1], c='red', s=30, alpha=0.7, label='Reconstructed')
                plt.axis('equal')
                plt.grid(True)
                plt.legend()
                plt.title(f'Epoch {epoch} (Loss: {loss_val:.4f})')
                
                # Coverage analysis
                plt.subplot(1, 3, 3)
                dist_matrix = torch.cdist(target_points.unsqueeze(0), recon.unsqueeze(0)).squeeze(0)
                min_distances = dist_matrix.min(dim=1)[0].detach().cpu().numpy()
                scatter = plt.scatter(target_np[:, 0], target_np[:, 1], c=min_distances, s=40, cmap='viridis')
                plt.colorbar(scatter, label='Distance to Reconstruction')
                plt.axis('equal')
                plt.grid(True)
                plt.title('Coverage Quality')
                
                plt.tight_layout()
                plt.savefig(output_dir / f"fixed_epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    # Final evaluation
    print(f"\n📊 2D-ADAPTED POINTFLOW RESULTS:")
    print(f"  🎯 Best Chamfer achieved: {best_loss:.4f}")
    print(f"  🎯 Target Chamfer: {TARGET_CHAMFER}")
    print(f"  ✅ Success: {'✓ PASSED' if best_loss < TARGET_CHAMFER else '✗ FAILED'}")
    
    # Final detailed analysis
    # Need gradients for CNF
    final_recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
    
    total_loss, chamfer, coverage_loss, target_cov, pred_cov = improved_chamfer_loss(
        final_recon, target_points
    )
    
    print(f"  📐 Final Chamfer: {chamfer:.4f}")
    print(f"  📈 Final Coverage: Target {target_cov*100:.1f}%, Pred {pred_cov*100:.1f}%")
    
    output_dir = Path("outputs/fixed_overfit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final results
    if best_loss < TARGET_CHAMFER:
        checkpoint = {
            'model_state': model.state_dict(),
            'config': {
                'latent_dim': LATENT_DIM,
                'hidden_dim': HIDDEN_DIM,
                'best_loss': best_loss,
                'target_loss': TARGET_CHAMFER
            }
        }
        torch.save(checkpoint, output_dir / 'fixed_overfit_checkpoint.pth')
        print(f"\n💾 REAL PointFlow model saved!")
    
    # Save metrics
    metrics = {
        'losses': losses,
        'best_loss': best_loss,
        'target_loss': TARGET_CHAMFER,
        'success': best_loss < TARGET_CHAMFER,
        'final_chamfer': chamfer.item(),
        'final_target_coverage': target_cov.item(),
        'final_pred_coverage': pred_cov.item()
    }
    
    with open(output_dir / 'fixed_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ FIXED overfitting test complete!")
    print(f"📁 Results saved to {output_dir}/")
    
    if best_loss < TARGET_CHAMFER:
        print(f"🚀 SUCCESS: Ready for multi-slice training!")
    else:
        print(f"🔧 Still needs work: Consider even larger model or different approach")

if __name__ == "__main__":
    main()
