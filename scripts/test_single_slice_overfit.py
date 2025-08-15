#!/usr/bin/env python3
"""
Single Slice Overfitting Test for PointFlow2D - FIXED VERSION
This script tests if the architecture can learn to perfectly reconstruct a single slice.
Includes critical fixes for stability and progress monitoring.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import warnings

from models.pointflow2d_final import PointFlow2DVAE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_single_slice(data_dir: Path, slice_name: str = None):
    """Load a single slice for overfitting test."""
    data_dir = Path(data_dir)
    
    if slice_name:
        slice_path = data_dir / slice_name
    else:
        # Find first .npy file
        npy_files = list(data_dir.glob("*.npy"))
        if not npy_files:
            raise ValueError(f"No .npy files found in {data_dir}")
        slice_path = npy_files[0]
    
    logger.info(f"Loading slice: {slice_path}")
    points = np.load(slice_path)
    
    # Convert to tensor and ensure it's 2D
    points_tensor = torch.from_numpy(points).float()
    if points_tensor.dim() == 2:
        points_tensor = points_tensor.unsqueeze(0)  # Add batch dimension
    
    # Ensure it's (batch, num_points, 2)
    if points_tensor.shape[-1] != 2:
        raise ValueError(f"Expected 2D points, got shape {points_tensor.shape}")
    
    logger.info(f"Loaded slice shape: {points_tensor.shape}")
    return points_tensor, slice_path.stem


def visualize_progress(original, reconstructed, epoch, save_dir):
    """Visualize original vs reconstructed points."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    orig_np = original[0].cpu().numpy()
    ax1.scatter(orig_np[:, 0], orig_np[:, 1], alpha=0.6, s=10)
    ax1.set_title(f'Original Slice ({orig_np.shape[0]} points)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Reconstructed
    recon_np = reconstructed[0].detach().cpu().numpy()
    ax2.scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.6, s=10)
    ax2.set_title(f'Reconstructed (Epoch {epoch})')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Make axes same scale
    all_points = np.concatenate([orig_np, recon_np], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    save_path = save_dir / f'reconstruction_epoch_{epoch:04d}.png'
    plt.savefig(save_path, dpi=100)
    plt.close()
    
    return save_path


def compute_chamfer_distance(x, y):
    """Compute Chamfer distance between two point sets."""
    # x: (batch, n, 2), y: (batch, m, 2)
    x_expanded = x.unsqueeze(2)  # (batch, n, 1, 2)
    y_expanded = y.unsqueeze(1)  # (batch, 1, m, 2)
    
    # Compute pairwise distances
    dist = torch.norm(x_expanded - y_expanded, dim=3)  # (batch, n, m)
    
    # Nearest neighbor distances
    min_dist_x_to_y, _ = torch.min(dist, dim=2)  # (batch, n)
    min_dist_y_to_x, _ = torch.min(dist, dim=1)  # (batch, m)
    
    # Chamfer distance
    chamfer = min_dist_x_to_y.mean() + min_dist_y_to_x.mean()
    return chamfer


def clip_grad_norm_(model, max_norm=5.0):
    """Clip gradients to prevent explosion."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def run_single_slice_overfit(
    data_dir: str,
    slice_name: str = None,
    epochs: int = 1000,
    latent_dim: int = 64,
    cnf_hidden_dim: int = 128,
    latent_cnf_hidden_dim: int = 128,
    learning_rate: float = 1e-4,  # REDUCED DEFAULT!
    device: str = 'auto',
    save_freq: int = 10,
    viz_freq: int = 50,
    grad_clip: float = 5.0,  # NEW: gradient clipping
    cnf_atol: float = 1e-4,  # INCREASED tolerance
    cnf_rtol: float = 1e-4   # INCREASED tolerance
):
    """Run single slice overfitting test with stability fixes."""
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / "single_slice_overfit" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Load single slice
    points, slice_id = load_single_slice(data_dir, slice_name)
    points = points.to(device)
    num_points = points.shape[1]
    
    # Normalize points to [-1, 1] for better training
    points_mean = points.mean(dim=1, keepdim=True)
    points_std = points.std(dim=1, keepdim=True).clamp(min=1e-6)
    points_normalized = (points - points_mean) / points_std
    
    logger.info(f"Points normalized: mean={points_mean.squeeze()}, std={points_std.squeeze()}")
    
    # Create model with increased ODE tolerances
    model = PointFlow2DVAE(
        input_dim=2,
        latent_dim=latent_dim,
        encoder_hidden_dim=256,
        cnf_hidden_dim=cnf_hidden_dim,
        latent_cnf_hidden_dim=latent_cnf_hidden_dim,
        use_latent_flow=True,
        cnf_atol=cnf_atol,  # INCREASED tolerance
        cnf_rtol=cnf_rtol,  # INCREASED tolerance
        force_cpu_ode=False  # Use GPU!
    ).to(device)
    
    info = model.get_model_info()
    logger.info(f"Model created: {info['total_parameters']:,} parameters")
    logger.info(f"Using learning rate: {learning_rate}")
    logger.info(f"Using gradient clipping: {grad_clip}")
    logger.info(f"ODE tolerances: atol={cnf_atol}, rtol={cnf_rtol}")
    
    # Create optimizer
    class Args:
        lr = learning_rate
        beta1 = 0.9
        beta2 = 0.999
        weight_decay = 0.
        optimizer = 'adam'
        momentum = 0.9
    
    args = Args()
    optimizer = model.make_optimizer(args)
    
    # Training loop with progress bar
    logger.info(f"Starting training for {epochs} epochs")
    losses = {'total': [], 'recon': [], 'prior': [], 'entropy': []}
    
    # Create progress bar
    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in pbar:
        try:
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with gradient computation
            batch_size = points_normalized.size(0)
            num_points = points_normalized.size(1)
            
            # Encode to get latent distribution
            z_mu, z_sigma = model.encoder(points_normalized)
            z = model.encoder.reparameterize(z_mu, z_sigma) if hasattr(model.encoder, 'reparameterize') else z_mu
            
            # Compute losses manually with error handling
            if model.use_latent_flow and model.latent_cnf is not None:
                w, delta_log_pw = model.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
                log_pw = model.standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
                delta_log_pw = delta_log_pw.view(batch_size, 1)
                log_pz = log_pw - delta_log_pw
            else:
                log_pz = model.standard_normal_logprob(z).view(batch_size, -1).sum(1, keepdim=True)
            
            # Compute reconstruction likelihood
            z_new = z.view(*z.size())
            z_new = z_new + (log_pz * 0.).mean()
            
            y, delta_log_py = model.point_cnf(
                points_normalized, z_new, torch.zeros(batch_size, num_points, 1).to(points_normalized)
            )
            log_py = model.standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
            log_px = log_py - delta_log_py
            
            # Compute entropy
            entropy = model.gaussian_entropy(z_sigma) if hasattr(model, 'gaussian_entropy') else torch.zeros(batch_size).to(z)
            
            # Losses
            entropy_loss = -entropy.mean() * model.entropy_weight
            recon_loss = -log_px.mean() * model.recon_weight
            prior_loss = -log_pz.mean() * model.prior_weight
            loss = entropy_loss + prior_loss + recon_loss
            
            # Backward with gradient clipping
            loss.backward()
            
            # CRITICAL: Clip gradients to prevent explosion
            clip_grad_norm_(model, grad_clip)
            
            # Optimizer step
            optimizer.step()
            
            # Track losses
            recon_nats = recon_loss.item() / float(points_normalized.size(1) * points_normalized.size(2))
            prior_nats = prior_loss.item() / float(model.latent_dim)
            
            losses['recon'].append(recon_nats)
            losses['prior'].append(prior_nats)
            losses['entropy'].append(entropy.mean().item())
            
            # Update progress bar
            pbar.set_postfix({
                'recon': f'{recon_nats:.4f}',
                'prior': f'{prior_nats:.4f}',
                'entropy': f'{entropy.mean().item():.2f}'
            })
            
            # Log progress every 10 epochs
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:4d} | Recon: {recon_nats:.6f} | "
                           f"Prior: {prior_nats:.6f} | Entropy: {entropy.mean().item():.3f}")
            
            # Visualize reconstruction
            if epoch % viz_freq == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    # Reconstruct
                    recon_normalized = model.reconstruct(points_normalized)
                    # Denormalize
                    recon = recon_normalized * points_std + points_mean
                    
                    # Compute Chamfer distance
                    chamfer = compute_chamfer_distance(points, recon)
                    logger.info(f"Epoch {epoch}: Chamfer distance = {chamfer:.6f}")
                    
                    # Visualize
                    viz_path = visualize_progress(points, recon, epoch, output_dir)
                    logger.info(f"Saved visualization: {viz_path}")
            
            # Save checkpoint
            if epoch % save_freq == 0 or epoch == epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'losses': losses,
                    'normalization': {
                        'mean': points_mean.cpu(),
                        'std': points_std.cpu()
                    }
                }
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch:04d}.pt'
                torch.save(checkpoint, checkpoint_path)
                
            # Early stopping if loss explodes
            if recon_nats > 1e6 or np.isnan(recon_nats):
                logger.error(f"Loss explosion detected at epoch {epoch}! Stopping training.")
                break
                
        except Exception as e:
            logger.error(f"Error at epoch {epoch}: {e}")
            logger.error("Attempting to continue training...")
            # Reset optimizer state if error occurs
            optimizer = model.make_optimizer(args)
            continue
    
    # Close progress bar
    pbar.close()
    
    # Plot loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(losses['recon'])
    axes[0].set_title('Reconstruction Loss (nats/dim)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_yscale('symlog')
    axes[0].grid(True)
    
    axes[1].plot(losses['prior'])
    axes[1].set_title('Prior Loss (nats/dim)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_yscale('symlog')
    axes[1].grid(True)
    
    axes[2].plot(losses['entropy'])
    axes[2].set_title('Entropy')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    plt.close()
    
    logger.info(f"Training complete! Results saved to {output_dir}")
    
    # Final evaluation
    with torch.no_grad():
        recon_normalized = model.reconstruct(points_normalized)
        recon = recon_normalized * points_std + points_mean
        final_chamfer = compute_chamfer_distance(points, recon)
        
        logger.info("\n" + "="*50)
        logger.info("FINAL RESULTS:")
        logger.info(f"  Final reconstruction loss: {losses['recon'][-1]:.6f} nats/dim")
        logger.info(f"  Final Chamfer distance: {final_chamfer:.6f}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Single slice overfitting test for PointFlow2D - FIXED VERSION")
    parser.add_argument("data_dir", type=str, help="Directory containing slice .npy files")
    parser.add_argument("--slice-name", type=str, default=None, 
                       help="Specific slice file name (e.g., 'DrivAer_F_D_WM_WW_1358_axis-x.npy')")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--cnf-hidden-dim", type=int, default=128, help="CNF hidden dimension")
    parser.add_argument("--latent-cnf-hidden-dim", type=int, default=128, 
                       help="Latent CNF hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 for stability)")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--save-freq", type=int, default=10, help="Save checkpoint frequency")
    parser.add_argument("--viz-freq", type=int, default=50, help="Visualization frequency")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping value")
    parser.add_argument("--cnf-atol", type=float, default=1e-4, help="ODE solver absolute tolerance")
    parser.add_argument("--cnf-rtol", type=float, default=1e-4, help="ODE solver relative tolerance")
    
    args = parser.parse_args()
    
    run_single_slice_overfit(
        data_dir=args.data_dir,
        slice_name=args.slice_name,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        cnf_hidden_dim=args.cnf_hidden_dim,
        latent_cnf_hidden_dim=args.latent_cnf_hidden_dim,
        learning_rate=args.lr,
        device=args.device,
        save_freq=args.save_freq,
        viz_freq=args.viz_freq,
        grad_clip=args.grad_clip,
        cnf_atol=args.cnf_atol,
        cnf_rtol=args.cnf_rtol
    )
