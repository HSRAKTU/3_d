#!/usr/bin/env python3
"""
Add visualization and analysis capabilities to VAE training.
To be integrated into train_vae_2d.py for better insights.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def save_training_visualizations(output_dir, training_log, epoch, model, val_loader, device):
    """Save comprehensive training visualizations"""
    
    # Create visualizations directory
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Loss curves
    if epoch % 50 == 0 or epoch == len(training_log) - 1:
        plot_loss_curves(training_log, viz_dir / f'loss_curves_epoch_{epoch}.png')
    
    # 2. Sample reconstructions
    if epoch % 100 == 0 or epoch == len(training_log) - 1:
        plot_reconstructions(model, val_loader, device, 
                           viz_dir / f'reconstructions_epoch_{epoch}.png')
    
    # 3. Latent space visualization
    if epoch % 200 == 0 or epoch == len(training_log) - 1:
        plot_latent_space(model, val_loader, device,
                         viz_dir / f'latent_space_epoch_{epoch}.png')
    
    # 4. Generation samples
    if epoch % 200 == 0 or epoch == len(training_log) - 1:
        plot_generations(model, device, 
                        viz_dir / f'generations_epoch_{epoch}.png')

def plot_loss_curves(training_log, save_path):
    """Plot all loss components over time"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = [log['epoch'] for log in training_log]
    
    # Total loss
    axes[0, 0].plot(epochs, [log['avg_loss'] for log in training_log])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_yscale('log')
    
    # Reconstruction loss
    axes[0, 1].plot(epochs, [log['recon'] for log in training_log])
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    
    # Prior loss
    axes[0, 2].plot(epochs, [log['prior'] for log in training_log])
    axes[0, 2].set_title('Prior Loss (KL)')
    axes[0, 2].set_xlabel('Epoch')
    
    # Chamfer distance
    axes[1, 0].plot(epochs, [log['val_chamfer'] for log in training_log])
    axes[1, 0].set_title('Validation Chamfer')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].axhline(y=0.1, color='r', linestyle='--', label='Target')
    
    # Log determinant
    axes[1, 1].plot(epochs, [log['log_det_mean'] for log in training_log])
    axes[1, 1].set_title('Log Determinant')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].axhline(y=0, color='k', linestyle='--')
    
    # Learning rate
    axes[1, 2].plot(epochs, [log['lr'] for log in training_log])
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_reconstructions(model, val_loader, device, save_path, num_samples=4):
    """Plot original vs reconstructed point clouds"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    
    with torch.no_grad():
        for i, (points, mask, _) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            points = points.to(device)
            mask = mask.to(device)
            points = points * mask.unsqueeze(-1)
            
            # Reconstruct
            z, _, _ = model.encode(points)
            recon = model.sample(z, points.shape[1])
            
            # Take first in batch
            orig = points[0].cpu().numpy()
            rec = recon[0].cpu().numpy()
            
            # Original
            axes[i, 0].scatter(orig[:, 0], orig[:, 1], s=1, c='blue')
            axes[i, 0].set_title(f'Original {i}')
            axes[i, 0].set_aspect('equal')
            
            # Reconstruction
            axes[i, 1].scatter(rec[:, 0], rec[:, 1], s=1, c='red')
            chamfer = compute_chamfer_distance(recon[:1], points[:1]).item()
            axes[i, 1].set_title(f'Recon {i} (CD: {chamfer:.4f})')
            axes[i, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def plot_latent_space(model, val_loader, device, save_path):
    """Visualize latent space distribution"""
    model.eval()
    
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for i, (points, mask, labels) in enumerate(val_loader):
            if i >= 10:  # Limit samples
                break
                
            points = points.to(device)
            mask = mask.to(device)
            points = points * mask.unsqueeze(-1)
            
            z, mu, logvar = model.encode(points)
            
            if mu is not None:
                all_z.append(mu.cpu().numpy())
            else:
                all_z.append(z.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_z = np.concatenate(all_z, axis=0)
    
    # Plot first 2 dimensions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Car ID')
    plt.xlabel('Latent Dim 0')
    plt.ylabel('Latent Dim 1')
    plt.title('Latent Space Visualization (First 2 Dims)')
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def plot_generations(model, device, save_path, num_samples=6):
    """Generate random samples from prior"""
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample from prior
            z = torch.randn(1, model.latent_dim).to(device)
            generated = model.sample(z, 500)  # 500 points
            
            gen_np = generated[0].cpu().numpy()
            axes[i].scatter(gen_np[:, 0], gen_np[:, 1], s=1, c='green')
            axes[i].set_title(f'Generated {i}')
            axes[i].set_aspect('equal')
            axes[i].set_xlim(-2, 2)
            axes[i].set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def compute_chamfer_distance(x, y):
    """Compute Chamfer distance"""
    x_expanded = x.unsqueeze(2)
    y_expanded = y.unsqueeze(1)
    distances = torch.norm(x_expanded - y_expanded, dim=3)
    min_dist_x_to_y = distances.min(dim=2)[0]
    min_dist_y_to_x = distances.min(dim=1)[0]
    return min_dist_x_to_y.mean() + min_dist_y_to_x.mean()

def save_final_analysis(output_dir, training_log, model, val_loader, device):
    """Save comprehensive final analysis"""
    
    analysis = {
        'training_summary': {
            'final_loss': training_log[-1]['avg_loss'],
            'best_chamfer': min([log['val_chamfer'] for log in training_log]),
            'best_epoch': min(training_log, key=lambda x: x['val_chamfer'])['epoch'],
            'final_recon_loss': training_log[-1]['recon'],
            'final_prior_loss': training_log[-1]['prior'],
            'final_log_det': training_log[-1]['log_det_mean'],
        },
        'loss_progression': {
            'chamfer_improvement': training_log[0]['val_chamfer'] - training_log[-1]['val_chamfer'],
            'loss_reduction': training_log[0]['avg_loss'] - training_log[-1]['avg_loss'],
            'converged': training_log[-1]['val_chamfer'] < 0.1,
        }
    }
    
    # Test reconstruction quality on full validation set
    model.eval()
    all_chamfers = []
    
    with torch.no_grad():
        for points, mask, _ in val_loader:
            points = points.to(device)
            mask = mask.to(device)
            points = points * mask.unsqueeze(-1)
            
            z, _, _ = model.encode(points)
            recon = model.sample(z, points.shape[1])
            
            for i in range(points.shape[0]):
                chamfer = compute_chamfer_distance(
                    recon[i:i+1], points[i:i+1]
                ).item()
                all_chamfers.append(chamfer)
    
    analysis['reconstruction_quality'] = {
        'mean_chamfer': np.mean(all_chamfers),
        'std_chamfer': np.std(all_chamfers),
        'min_chamfer': np.min(all_chamfers),
        'max_chamfer': np.max(all_chamfers),
        'median_chamfer': np.median(all_chamfers),
    }
    
    # Save analysis
    with open(output_dir / 'final_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Plot chamfer distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_chamfers, bins=50, alpha=0.7)
    plt.axvline(np.mean(all_chamfers), color='r', linestyle='--', label=f'Mean: {np.mean(all_chamfers):.4f}')
    plt.axvline(0.1, color='g', linestyle='--', label='Target: 0.1')
    plt.xlabel('Chamfer Distance')
    plt.ylabel('Count')
    plt.title('Reconstruction Quality Distribution')
    plt.legend()
    plt.savefig(output_dir / 'chamfer_distribution.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    print("This module provides visualization functions for VAE training.")
    print("Import and use in train_vae_2d.py for comprehensive insights.")
