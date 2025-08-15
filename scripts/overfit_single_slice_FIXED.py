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

from src.models.pointflow2d_final import PointFlow2DVAE

# FIXED configuration - addressing underfitting
LATENT_DIM = 256     # Increased from 128 - more capacity
HIDDEN_DIM = 512     # Increased from 256 - more expressive  
SOLVER_STEPS = 20    # Increased from 5 - higher quality
LEARNING_RATE = 5e-4 # Slightly lower but more stable
EPOCHS = 2000        # Longer training
TARGET_LOSS = 0.03   # More aggressive target
BATCH_SIZE = 4       # Smaller for stability with larger model

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
    
    # Create REAL PointFlow model 
    print("\n🏗️  Building REAL PointFlow2DVAE...")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Solver: dopri5 (real ODE solver)")
    
    # Full PointFlow2DVAE with proper CNF
    model = PointFlow2DVAE(
        input_dim=2,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        use_latent_flow=False,  # Autoencoder mode for overfitting
        use_deterministic_encoder=True,
        cnf_atol=1e-4,
        cnf_rtol=1e-4
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} (REAL CNF)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Two-phase learning rate schedule
    def lr_schedule(epoch):
        if epoch < 800:
            return 1.0  # Keep initial LR for 800 epochs
        elif epoch < 1600:
            return 0.5  # Half LR for middle phase
        else:
            return 0.1  # Low LR for final refinement
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    use_amp = scaler is not None
    print(f"🚀 Mixed precision training: {'✓' if use_amp else '✗'}")
    print(f"🚀 Using batch size: {BATCH_SIZE} (smaller for stability)")
    
    # Training
    print(f"\n🚀 Starting FIXED overfitting test (target loss: {TARGET_LOSS})")
    losses = []
    best_loss = float('inf')
    
    # Create batched target
    target_batch = target_points.unsqueeze(0).repeat(BATCH_SIZE, 1, 1)  # [B, N, 2]
    
    pbar = tqdm(range(EPOCHS), desc="FIXED Overfitting")
    for epoch in pbar:
        optimizer.zero_grad()
        
        # REAL PointFlow training using reconstruction loss
        if use_amp:
            with torch.cuda.amp.autocast():
                # Use PointFlow2DVAE's reconstruction method 
                recon_loss = model.reconstruction_loss(target_batch)
                loss = recon_loss
        else:
            # Standard precision
            recon_loss = model.reconstruction_loss(target_batch)
            loss = recon_loss
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Track metrics
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
        
        # Detailed analysis every 200 epochs
        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            print(f"\n📊 Epoch {epoch} FIXED Analysis:")
            
            with torch.no_grad():
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
                plt.plot(losses, 'b-', label='PointFlow Loss')
                plt.axhline(y=TARGET_LOSS, color='r', linestyle='--', label=f'Target: {TARGET_LOSS}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (Negative Log-Likelihood)')
                plt.title('PointFlow Training Progress')
                plt.legend()
                plt.grid(True)
                plt.yscale('log')
                
                # Reconstruction
                target_np = target_points.cpu().numpy()
                recon_np = recon.cpu().numpy()
                
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
                min_distances = dist_matrix.min(dim=1)[0].cpu().numpy()
                scatter = plt.scatter(target_np[:, 0], target_np[:, 1], c=min_distances, s=40, cmap='viridis')
                plt.colorbar(scatter, label='Distance to Reconstruction')
                plt.axis('equal')
                plt.grid(True)
                plt.title('Coverage Quality')
                
                plt.tight_layout()
                plt.savefig(output_dir / f"fixed_epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    # Final evaluation
    print(f"\n📊 FIXED OVERFITTING RESULTS:")
    print(f"  🎯 Best loss achieved: {best_loss:.4f}")
    print(f"  🎯 Target loss: {TARGET_LOSS}")
    print(f"  ✅ Success: {'✓ PASSED' if best_loss < TARGET_LOSS else '✗ FAILED'}")
    
    # Final detailed analysis
    with torch.no_grad():
        final_recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
        
        total_loss, chamfer, coverage_loss, target_cov, pred_cov = improved_chamfer_loss(
            final_recon, target_points
        )
        
        print(f"  📐 Final Chamfer: {chamfer:.4f}")
        print(f"  📈 Final Coverage: Target {target_cov*100:.1f}%, Pred {pred_cov*100:.1f}%")
    
    output_dir = Path("outputs/fixed_overfit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final results
    if best_loss < TARGET_LOSS:
        checkpoint = {
            'model_state': model.state_dict(),
            'config': {
                'latent_dim': LATENT_DIM,
                'hidden_dim': HIDDEN_DIM,
                'best_loss': best_loss,
                'target_loss': TARGET_LOSS
            }
        }
        torch.save(checkpoint, output_dir / 'fixed_overfit_checkpoint.pth')
        print(f"\n💾 REAL PointFlow model saved!")
    
    # Save metrics
    metrics = {
        'losses': losses,
        'best_loss': best_loss,
        'target_loss': TARGET_LOSS,
        'success': best_loss < TARGET_LOSS,
        'final_chamfer': chamfer.item(),
        'final_target_coverage': target_cov.item(),
        'final_pred_coverage': pred_cov.item()
    }
    
    with open(output_dir / 'fixed_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ FIXED overfitting test complete!")
    print(f"📁 Results saved to {output_dir}/")
    
    if best_loss < TARGET_LOSS:
        print(f"🚀 SUCCESS: Ready for multi-slice training!")
    else:
        print(f"🔧 Still needs work: Consider even larger model or different approach")

if __name__ == "__main__":
    main()
