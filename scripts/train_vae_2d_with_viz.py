#!/usr/bin/env python3
"""
Enhanced VAE training with visualizations.
Quick modification to add to train_vae_2d.py
"""

# Add these imports to train_vae_2d.py
import matplotlib
matplotlib.use('Agg')  # For headless plotting
import matplotlib.pyplot as plt

# Add this function to train_vae_2d.py
def save_epoch_visualization(epoch, model, val_loader, device, output_dir):
    """Save visualizations during training"""
    if epoch % 50 != 0 and epoch != 0:
        return
        
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    model.eval()
    
    # Get one batch for visualization
    for points, mask, _ in val_loader:
        points = points.to(device)[:4]  # Take 4 samples
        mask = mask.to(device)[:4]
        points = points * mask.unsqueeze(-1)
        break
    
    with torch.no_grad():
        # Reconstruct
        z, _, _ = model.encode(points)
        recon = model.sample(z, points.shape[1])
        
        # Plot
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(4):
            # Original
            orig = points[i].cpu().numpy()
            axes[0, i].scatter(orig[:, 0], orig[:, 1], s=1, c='blue', alpha=0.5)
            axes[0, i].set_title(f'Original {i}')
            axes[0, i].set_aspect('equal')
            axes[0, i].set_xlim(-2, 2)
            axes[0, i].set_ylim(-1, 1.5)
            
            # Reconstruction
            rec = recon[i].cpu().numpy()
            axes[1, i].scatter(rec[:, 0], rec[:, 1], s=1, c='red', alpha=0.5)
            chamfer = compute_chamfer_distance(recon[i:i+1], points[i:i+1]).item()
            axes[1, i].set_title(f'Recon (CD: {chamfer:.4f})')
            axes[1, i].set_aspect('equal')
            axes[1, i].set_xlim(-2, 2)
            axes[1, i].set_ylim(-1, 1.5)
        
        plt.suptitle(f'Epoch {epoch} Reconstructions')
        plt.tight_layout()
        plt.savefig(viz_dir / f'recon_epoch_{epoch:04d}.png', dpi=150)
        plt.close()
        
    model.train()

# Add to main training loop (after validation):
save_epoch_visualization(epoch, model, val_loader, device, output_dir)
