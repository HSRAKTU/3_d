#!/usr/bin/env python3
"""
10-Car Training Using EXACT SAME Architecture as Single Slice Success
This script uses PointFlow2DAdaptedVAE - the proven architecture
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
from scipy.spatial import cKDTree
sys.path.append('.')

# EXACT SAME MODEL AS SINGLE SLICE SUCCESS
from src.models.pointflow2d_adapted import PointFlow2DAdaptedVAE
from src.training.dataset import SliceDataset, collate_variable_length
from src.data.loader import SliceDataLoader
from torch.utils.data import DataLoader, random_split, Subset

# Import the EXACT SAME helper functions from stable script
from overfit_single_slice_STABLE import (
    compute_chamfer_distance,
    improved_chamfer_loss
)

# Configuration - adapted for multi-slice but same architecture
LATENT_DIM = 128  # SAME as single slice
LEARNING_RATE = 5e-4  # SAME as single slice
MIN_LR = 1e-4  # SAME as single slice
EPOCHS = 300  # More epochs with early stopping
TARGET_CHAMFER = 0.1  # Less strict (multiple shapes)
BATCH_SIZE = 32  # Increased from 8 (need variety)
GRADIENT_CLIP = 1.0  # SAME as single slice
WEIGHT_DECAY = 1e-4  # SAME as single slice
VAL_SPLIT = 0.1  # 10% validation
NUM_CARS = 10
EARLY_STOP_PATIENCE = 30  # Stop if no improvement for 30 epochs

def main():
    """Main training function for 10 cars"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/10cars_stable_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("📊 Loading 10-car dataset...")
    
    # First get the car IDs and limit to NUM_CARS
    loader = SliceDataLoader("data/training_dataset")
    available_car_ids = loader.get_car_ids()
    selected_car_ids = available_car_ids[:NUM_CARS]  # Take first 10 cars
    
    print(f"   Selected cars: {selected_car_ids[:3]}... (showing first 3 of {len(selected_car_ids)})")
    
    # Load dataset with only the selected cars
    dataset = SliceDataset(
        data_directory="data/training_dataset",
        car_ids=selected_car_ids,  # Pass specific car IDs
        normalize=True,
        max_points=1000,  # Filter outliers
        min_points=10
    )
    
    print(f"   Using {len(dataset)} slices from {NUM_CARS} cars")
    
    # Split dataset
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"   Train: {len(train_dataset)} slices, Val: {len(val_dataset)} slices")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_variable_length
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_variable_length
    )
    
    # EXACT SAME MODEL CONFIGURATION AS SINGLE SLICE
    model = PointFlow2DAdaptedVAE(
        input_dim=2,
        latent_dim=LATENT_DIM,
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,
        solver='euler',
        solver_steps=10,
        use_deterministic_encoder=True
    ).to(device)
    
    # Print model info to confirm it's the same
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # Optimizer - same as single slice
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler - same type as single slice
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_chamfer': [],
        'learning_rate': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n🚀 Starting training...")
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_losses = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Get batch data
                points = batch['points'].to(device)
                mask = batch['mask'].to(device)
                num_points = batch['num_points']
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - EXACT SAME AS SINGLE SLICE
                z = model.encode(points)
                y, log_det = model.point_cnf(points, z, reverse=False)
                
                # Log probability under standard normal
                log_py = torch.distributions.Normal(0, 1).log_prob(y)
                log_py = log_py.view(points.shape[0], -1).sum(1, keepdim=True)
                
                # Add log determinant
                log_px = log_py + log_det.view(points.shape[0], -1).sum(1, keepdim=True)
                
                # Loss is negative log likelihood
                loss = -log_px.mean()
                
                # Add L2 regularization on CNF output
                output_reg = 0.01 * (y ** 2).mean()
                loss = loss + output_reg
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
                
                # Optimizer step
                optimizer.step()
                
                # Record loss
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_losses = []
        val_chamfers = []
        
        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(device)
                mask = batch['mask'].to(device)
                num_points = batch['num_points']
                
                # Reconstruction loss - SAME AS TRAINING
                z = model.encode(points)
                y, log_det = model.point_cnf(points, z, reverse=False)
                
                log_py = torch.distributions.Normal(0, 1).log_prob(y)
                log_py = log_py.view(points.shape[0], -1).sum(1, keepdim=True)
                log_px = log_py + log_det.view(points.shape[0], -1).sum(1, keepdim=True)
                
                val_loss = -log_px.mean() + 0.01 * (y ** 2).mean()
                val_losses.append(val_loss.item())
                
                # Compute Chamfer distance for first item in batch
                if len(val_chamfers) < 10:  # Sample a few
                    n_points = num_points[0].item()
                    target = points[0, :n_points]
                    
                    # Reconstruct single item
                    recon = model.reconstruct(points[0:1])
                    recon = recon[0, :n_points]
                    
                    chamfer = compute_chamfer_distance(recon, target)
                    val_chamfers.append(chamfer)
        
        # Record epoch stats
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_chamfer = np.mean(val_chamfers) if val_chamfers else float('inf')
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_chamfer'].append(avg_val_chamfer)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Chamfer: {avg_val_chamfer:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_chamfer': avg_val_chamfer,
                'config': {
                    'latent_dim': LATENT_DIM,
                    'num_cars': NUM_CARS,
                    'architecture': 'PointFlow2DAdaptedVAE'
                }
            }, output_dir / 'best_model.pth')
            print(f"   ✅ Saved best model (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter > EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
        
        # Visualize sample every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Get a validation sample
                val_batch = next(iter(val_loader))
                points = val_batch['points'][0:1].to(device)
                mask = val_batch['mask'][0:1].to(device)
                num_points = val_batch['num_points'][0:1]
                
                # Reconstruct
                recon = model.reconstruct(points, mask, num_points)
                
                # Extract actual points
                n_points = num_points[0].item()
                target_np = points[0, :n_points].cpu().numpy()
                recon_np = recon[0, :n_points].cpu().numpy()
                
                # Plot
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                ax1.scatter(target_np[:, 0], target_np[:, 1], alpha=0.5, s=1)
                ax1.set_title(f'Original (Epoch {epoch+1})')
                ax1.axis('equal')
                
                ax2.scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.5, s=1, color='red')
                ax2.set_title('Reconstructed')
                ax2.axis('equal')
                
                ax3.scatter(target_np[:, 0], target_np[:, 1], alpha=0.5, s=1, label='Original')
                ax3.scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.5, s=1, color='red', label='Reconstructed')
                ax3.set_title(f'Overlay (Chamfer: {avg_val_chamfer:.4f})')
                ax3.axis('equal')
                ax3.legend()
                
                plt.tight_layout()
                plt.savefig(output_dir / f'epoch_{epoch+1:04d}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        # Update learning rate
        scheduler.step()
    
    # Save final results
    print("\n📊 Saving final results...")
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Chamfer distance
    axes[0, 1].plot(history['val_chamfer'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Chamfer Distance')
    axes[0, 1].set_title('Validation Chamfer Distance')
    axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    
    # Final summary
    axes[1, 1].text(0.1, 0.8, f"Final Results:", fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.6, f"Train Loss: {history['train_loss'][-1]:.4f}", fontsize=12)
    axes[1, 1].text(0.1, 0.5, f"Val Loss: {history['val_loss'][-1]:.4f}", fontsize=12)
    axes[1, 1].text(0.1, 0.4, f"Val Chamfer: {history['val_chamfer'][-1]:.4f}", fontsize=12)
    axes[1, 1].text(0.1, 0.3, f"Best Val Loss: {best_val_loss:.4f}", fontsize=12)
    axes[1, 1].text(0.1, 0.2, f"Total Epochs: {len(history['train_loss'])}", fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Training complete! Results saved to: {output_dir}")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Final validation Chamfer: {history['val_chamfer'][-1]:.4f}")
    
    # Only do comprehensive evaluation if model converged well
    if best_val_loss < 0.5:  # Reasonable threshold
        print("\n✅ Model converged well! Running comprehensive evaluation...")
        
        # Load best model
        checkpoint = torch.load(output_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # Sample diverse slices - different cars, different positions
        sample_indices = [0, 10, 20, 40, 60, 80]  # Spread across dataset
        
        for plot_idx, data_idx in enumerate(sample_indices):
            # Get specific validation sample
            val_item = val_dataset[data_idx % len(val_dataset)]
            points = val_item['points'].unsqueeze(0).to(device)
            num_points = val_item['num_points'].unsqueeze(0)
            car_id = val_item['car_id']
            slice_idx = val_item['slice_idx']
            
            # Handle variable length
            if hasattr(val_dataset, 'dataset'):  # If using Subset
                mask = torch.ones_like(points[..., 0])
                mask[0, num_points[0]:] = 0
            else:
                mask = torch.ones_like(points[..., 0])
            
            # Reconstruct
            with torch.no_grad():
                recon = model.reconstruct(points, mask, num_points)
            
            n_points = num_points[0].item()
            target_np = points[0, :n_points].cpu().numpy()
            recon_np = recon[0, :n_points].cpu().numpy()
            
            # Compute metrics
            chamfer = compute_chamfer_distance(recon[0, :n_points], points[0, :n_points])
            _, _, _, target_cov, pred_cov = improved_chamfer_loss(
                recon[0, :n_points], points[0, :n_points]
            )
            
            # Plot row: Original, Reconstructed, Overlay, Error Heatmap
            row = plot_idx
            
            # Original
            ax1 = plt.subplot(6, 4, row*4 + 1)
            ax1.scatter(target_np[:, 0], target_np[:, 1], alpha=0.6, s=2, c='blue')
            ax1.set_title(f'Car {car_id}, Slice {slice_idx}')
            ax1.axis('equal')
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
            
            # Reconstructed
            ax2 = plt.subplot(6, 4, row*4 + 2)
            ax2.scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.6, s=2, c='red')
            ax2.set_title(f'Chamfer: {chamfer:.4f}')
            ax2.axis('equal')
            ax2.set_xlim(-1.5, 1.5)
            ax2.set_ylim(-1.5, 1.5)
            
            # Overlay
            ax3 = plt.subplot(6, 4, row*4 + 3)
            ax3.scatter(target_np[:, 0], target_np[:, 1], alpha=0.5, s=2, c='blue', label='Original')
            ax3.scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.5, s=2, c='red', label='Recon')
            ax3.set_title(f'Coverage: T={target_cov:.1%} P={pred_cov:.1%}')
            ax3.axis('equal')
            ax3.set_xlim(-1.5, 1.5)
            ax3.set_ylim(-1.5, 1.5)
            if plot_idx == 0:
                ax3.legend()
            
            # Point-wise error
            ax4 = plt.subplot(6, 4, row*4 + 4)
            # Compute nearest neighbor distances
            tree = cKDTree(target_np)
            distances, _ = tree.query(recon_np)
            scatter = ax4.scatter(recon_np[:, 0], recon_np[:, 1], 
                                 c=distances, cmap='hot', s=2, 
                                 vmin=0, vmax=0.1)
            ax4.set_title('Point-wise Error')
            ax4.axis('equal')
            ax4.set_xlim(-1.5, 1.5)
            ax4.set_ylim(-1.5, 1.5)
            if plot_idx == 0:
                plt.colorbar(scatter, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_evaluation.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics
        summary = {
            'best_val_loss': float(best_val_loss),
            'best_epoch': int(checkpoint['epoch']),
            'total_epochs_trained': len(history['train_loss']),
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'final_val_chamfer': float(history['val_chamfer'][-1]),
            'model_config': checkpoint['config']
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Comprehensive evaluation saved!")
        print(f"   Best model checkpoint: {output_dir}/best_model.pth")
        print(f"   Visualizations: {output_dir}/comprehensive_evaluation.png")
        print(f"   Summary: {output_dir}/summary.json")
    else:
        print(f"\n⚠️  Model did not converge well (best_val_loss={best_val_loss:.4f})")
        print(f"   Best model still saved at: {output_dir}/best_model.pth")
    
    print(f"\n🎉 Training script complete!")

if __name__ == "__main__":
    main()
