#!/usr/bin/env python3
"""
STABLE Single Slice Overfitting - With stability fixes
Prevents the scale explosion seen after epoch 800
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

def improved_chamfer_loss(pred, target, coverage_weight=0.3):
    """
    Improved loss function with coverage metrics
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
    
    return total_loss.item(), chamfer.item(), coverage_loss.item(), target_covered.item(), pred_coverage.item()

# STABLE configuration with tweaks
LATENT_DIM = 128
LEARNING_RATE = 5e-4  # Lower initial LR for stability
MIN_LR = 1e-4  # Don't go too low! This prevents instability
EPOCHS = 1000
TARGET_CHAMFER = 0.05  # More realistic target
BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 100  # Stop if no improvement
GRADIENT_CLIP = 1.0  # More aggressive clipping
WEIGHT_DECAY = 1e-4  # Higher weight decay for regularization

def load_single_slice():
    """Load a single slice for overfitting test"""
    possible_paths = [
        Path("data/single_slice_test.npy"),
        Path("../data/single_slice_test.npy"),
    ]
    
    for path in possible_paths:
        if path.exists():
            data = np.load(path, allow_pickle=True)
            
            if isinstance(data, dict) and 'points' in data:
                points = data['points']
            elif isinstance(data, np.ndarray):
                if data.ndim == 2:
                    points = data
                else:
                    points = data[0] if data.shape[0] > 0 else data
            else:
                points = np.array(data)
            
            if isinstance(points, torch.Tensor):
                points = points.numpy()
            
            points = np.array(points, dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(-1, 2)
            
            return points, path
    
    raise FileNotFoundError(f"Could not find single_slice_test.npy")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load data
    print("\n📂 Loading single slice...")
    target_points, data_path = load_single_slice()
    
    # Center and scale
    center = target_points.mean(axis=0)
    scale = target_points.std()
    target_points = (target_points - center) / scale
    
    target_points = torch.from_numpy(target_points).float()
    target_points = target_points.to(device)
    num_points = target_points.shape[0]
    print(f"✓ Loaded slice with {num_points} points from: {data_path}")
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}], Scale: {scale:.3f}")
    
    # Create STABLE 2D-ADAPTED PointFlow model 
    print("\n🏗️  Building STABLE 2D-ADAPTED PointFlow...")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  Gradient clipping: {GRADIENT_CLIP}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Min LR: {MIN_LR}")
    
    # Create model with modified settings
    model = PointFlow2DAdaptedVAE(
        input_dim=2,
        latent_dim=LATENT_DIM,
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,
        solver='euler',
        solver_steps=10,
        use_deterministic_encoder=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer with higher weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Better learning rate schedule - don't go too low!
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=MIN_LR
    )
    
    # Training
    print(f"\n🚀 Starting STABLE training (target Chamfer: {TARGET_CHAMFER})")
    print(f"🚀 Early stopping patience: {EARLY_STOP_PATIENCE} epochs")
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Create batched target
    target_batch = target_points.unsqueeze(0).repeat(BATCH_SIZE, 1, 1)  # [B, N, 2]
    
    pbar = tqdm(range(EPOCHS), desc="STABLE PointFlow")
    for epoch in pbar:
        # Forward pass
        step = epoch
        
        # Override gradient clipping in model
        optimizer.zero_grad()
        
        # Get loss directly
        B, N, D = target_batch.shape
        z = model.encode(target_batch)
        y, log_det = model.point_cnf(target_batch, z, reverse=False)
        
        # Log probability under standard normal
        log_py = -0.5 * (y ** 2).sum(dim=-1) - 0.5 * D * np.log(2 * np.pi)
        log_py = log_py.sum(dim=1, keepdim=True)
        
        # Log probability of data
        log_px = log_py + log_det
        
        # Loss is negative log likelihood
        loss = -log_px.mean()
        
        # Add L2 regularization on CNF output to prevent explosion
        output_reg = 0.01 * (y ** 2).mean()
        loss = loss + output_reg
        
        # Backward with aggressive gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        optimizer.step()
        
        scheduler.step()
        
        # Track loss
        pointflow_loss = loss.item()
        
        # Compute ACTUAL reconstruction quality every 10 epochs
        if epoch % 10 == 0:
            recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
            
            # Check for explosion
            recon_scale = recon.abs().max().item()
            if recon_scale > 10.0:
                print(f"\n⚠️ WARNING: Reconstruction scale explosion detected: {recon_scale:.2f}")
                print("Stopping training to prevent instability...")
                break
            
            chamfer_dist = compute_chamfer_distance(target_points, recon)
            loss_val = chamfer_dist
            
            # Early stopping check
            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                Path('outputs').mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'outputs/best_model.pth')
            else:
                patience_counter += 1
        else:
            loss_val = losses[-1] if losses else float('inf')
        
        losses.append(loss_val)
        
        pbar.set_postfix({
            'chamfer': f'{loss_val:.4f}',
            'best': f'{best_loss:.4f}',
            'pf_loss': f'{pointflow_loss:.3f}',
            'lr': f'{scheduler.get_last_lr()[0]:.1e}',
            'patience': patience_counter
        })
        
        # Early stopping
        if loss_val < TARGET_CHAMFER:
            print(f"\n🎯 Target Chamfer distance reached at epoch {epoch}!")
            break
            
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⏰ Early stopping triggered at epoch {epoch}")
            print(f"   Best loss was {best_loss:.4f} at epoch {best_epoch}")
            break
        
        # Detailed analysis and visualization every 200 epochs
        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            print(f"\n📊 Epoch {epoch} Analysis:")
            
            # Get reconstruction and metrics
            recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
            
            # Detailed metrics
            total_loss, chamfer, coverage_loss, target_cov, pred_cov = improved_chamfer_loss(
                recon, target_points
            )
            
            print(f"  🎯 Total Loss: {total_loss:.4f} (Chamfer: {chamfer:.4f}, Coverage: {coverage_loss:.4f})")
            print(f"  📐 Point Coverage: Target {target_cov*100:.1f}%, Pred {pred_cov*100:.1f}%")
            print(f"  📈 Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            print(f"  🔧 Scale check: Max coord = {recon.abs().max().item():.2f}")
            
            # Save visualization
            if epoch % 400 == 0 or epoch == EPOCHS - 1:
                plt.figure(figsize=(15, 5))
                
                # Loss curve
                plt.subplot(1, 3, 1)
                plt.plot(losses, 'b-', label='Chamfer Distance')
                plt.axhline(y=TARGET_CHAMFER, color='r', linestyle='--', label=f'Target: {TARGET_CHAMFER}')
                plt.xlabel('Epoch')
                plt.ylabel('Chamfer Distance')
                plt.title('Stable Training Progress')
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
                plt.title(f'Epoch {epoch} (Chamfer: {chamfer:.4f})')
                
                # Coverage quality
                plt.subplot(1, 3, 3)
                dist_matrix = torch.cdist(target_points.unsqueeze(0), recon.unsqueeze(0)).squeeze(0)
                min_distances = dist_matrix.min(dim=1)[0].detach().cpu().numpy()
                scatter = plt.scatter(target_np[:, 0], target_np[:, 1], c=min_distances, s=40, cmap='viridis')
                plt.colorbar(scatter, label='Distance to Reconstruction')
                plt.axis('equal')
                plt.grid(True)
                plt.title('Coverage Quality')
                
                plt.tight_layout()
                output_dir = Path("outputs/stable_overfit")
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f"stable_epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    # Final evaluation
    print(f"\n📊 STABLE TRAINING RESULTS:")
    print(f"  🎯 Best Chamfer achieved: {best_loss:.4f} at epoch {best_epoch}")
    print(f"  🎯 Target Chamfer: {TARGET_CHAMFER}")
    print(f"  ✅ Success: {'✓ PASSED' if best_loss < TARGET_CHAMFER else '✗ FAILED'}")
    
    # Load best model
    if Path('outputs/best_model.pth').exists():
        print("\n💾 Loading best model checkpoint...")
        checkpoint = torch.load('outputs/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final analysis with best model
        final_recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
        
        # Get all metrics
        total_loss, chamfer, coverage_loss, target_cov, pred_cov = improved_chamfer_loss(
            final_recon, target_points
        )
        
        print(f"\n📊 BEST MODEL ANALYSIS:")
        print(f"  📐 Final Chamfer: {chamfer:.4f}")
        print(f"  📈 Final Coverage: Target {target_cov*100:.1f}%, Pred {pred_cov*100:.1f}%")
        print(f"  🔧 Max coordinate: {final_recon.abs().max().item():.2f}")
        
        # Save final results
        output_dir = Path("outputs/stable_overfit")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_results = {
            'best_epoch': best_epoch,
            'best_chamfer': best_loss,
            'final_chamfer': chamfer,
            'target_coverage': target_cov,
            'pred_coverage': pred_cov,
            'max_coordinate': final_recon.abs().max().item(),
            'total_params': total_params,
            'config': {
                'latent_dim': LATENT_DIM,
                'learning_rate': LEARNING_RATE,
                'min_lr': MIN_LR,
                'gradient_clip': GRADIENT_CLIP,
                'weight_decay': WEIGHT_DECAY,
                'batch_size': BATCH_SIZE,
                'epochs': epoch + 1
            }
        }
        
        with open(output_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save final visualization
        create_final_visualization(target_points, final_recon, losses, best_epoch, output_dir)
        
        if chamfer < TARGET_CHAMFER:
            print("\n🎉 SUCCESS: Model achieved target performance!")
            print("💾 Model checkpoint saved to outputs/best_model.pth")
        else:
            print("\n🔧 Model needs more tuning to reach target performance")
    
    print("\n✅ STABLE overfitting test complete!")
    print("📁 Results saved to outputs/stable_overfit/")

def create_final_visualization(target_points, recon, losses, best_epoch, output_dir):
    """Create comprehensive final visualization"""
    plt.figure(figsize=(20, 5))
    
    # Loss progression
    plt.subplot(1, 4, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best: {best_epoch}')
    plt.axhline(y=TARGET_CHAMFER, color='r', linestyle='--', label=f'Target: {TARGET_CHAMFER}')
    plt.xlabel('Epoch')
    plt.ylabel('Chamfer Distance')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Best reconstruction
    target_np = target_points.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()
    
    plt.subplot(1, 4, 2)
    plt.scatter(target_np[:, 0], target_np[:, 1], c='blue', s=30, alpha=0.7, label='Target')
    plt.scatter(recon_np[:, 0], recon_np[:, 1], c='red', s=30, alpha=0.7, label='Reconstructed')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Final Reconstruction')
    
    # Overlay
    plt.subplot(1, 4, 3)
    plt.scatter(target_np[:, 0], target_np[:, 1], c='blue', s=40, alpha=0.3)
    plt.scatter(recon_np[:, 0], recon_np[:, 1], c='red', s=15, alpha=0.8)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Overlay (Blue: Target, Red: Recon)')
    
    # Error heatmap
    plt.subplot(1, 4, 4)
    dist_matrix = torch.cdist(target_points.unsqueeze(0), recon.unsqueeze(0)).squeeze(0)
    min_distances = dist_matrix.min(dim=1)[0].detach().cpu().numpy()
    scatter = plt.scatter(target_np[:, 0], target_np[:, 1], c=min_distances, s=40, cmap='plasma')
    plt.colorbar(scatter, label='Error')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Point-wise Error')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_summary.png', dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
