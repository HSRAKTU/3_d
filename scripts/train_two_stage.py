#!/usr/bin/env python3
"""
Two-stage training for PointFlow2D following the original PointFlow strategy.

Stage 1: Train as deterministic autoencoder (reconstruction only)
Stage 2: Enable full VAE with latent flow
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import PointFlow2DVAE
from models.encoder import gaussian_entropy
from models.pointflow2d_final import standard_normal_logprob

logger = logging.getLogger(__name__)


def load_single_slice(data_dir: str, slice_name: str = None):
    """Load a single slice for testing."""
    data_path = Path(data_dir)
    
    if slice_name:
        slice_path = data_path / slice_name
        if not slice_path.exists():
            raise FileNotFoundError(f"Slice not found: {slice_path}")
    else:
        # Find first .npy file
        npy_files = list(data_path.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {data_path}")
        slice_path = npy_files[0]
        slice_name = slice_path.name
    
    # Load slice
    slice_data = np.load(slice_path)
    if slice_data.ndim == 2:
        slice_data = slice_data[np.newaxis, :, :]  # Add batch dimension
    
    points = torch.tensor(slice_data, dtype=torch.float32)
    logger.info(f"Loaded slice: {slice_path}")
    logger.info(f"Slice shape: {points.shape}")
    
    return points, slice_name


class TwoStageTrainer:
    """Handles two-stage training for PointFlow2D."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.stage = 1
        
    def train_stage1(self, points, optimizer, epochs=500):
        """Stage 1: Deterministic autoencoder training."""
        logger.info("="*50)
        logger.info("STAGE 1: Autoencoder Training (Reconstruction Only)")
        logger.info("="*50)
        
        losses = []
        pbar = tqdm(range(epochs), desc="Stage 1")
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Forward pass - Stage 1 style
            batch_size = points.size(0)
            num_points = points.size(1)
            
            # Deterministic encoding (only use mu)
            z_mu, z_sigma = self.model.encoder(points)
            z = z_mu  # Deterministic!
            
            # Skip latent CNF - no prior modeling
            log_pz = torch.zeros(batch_size, 1).to(z)
            
            # Point CNF: transform points to Gaussian
            y, delta_log_py = self.model.point_cnf(
                points, z, torch.zeros(batch_size, num_points, 1).to(points)
            )
            
            # Reconstruction likelihood
            log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
            log_px = log_py - delta_log_py
            
            # ONLY reconstruction loss for Stage 1
            loss = -log_px.mean()
            
            # Check for numerical issues
            if torch.isnan(loss) or loss.item() > 1e6:
                logger.warning(f"Numerical issue at epoch {epoch}: loss={loss.item()}")
                # Skip this step
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Visualize periodically
            if (epoch + 1) % 50 == 0:
                self.visualize_reconstruction(points, z, epoch, stage=1)
        
        return losses
    
    def train_stage2(self, points, optimizer, epochs=500):
        """Stage 2: Full VAE with latent flow."""
        logger.info("="*50)
        logger.info("STAGE 2: Full Generative Model (VAE + Latent Flow)")
        logger.info("="*50)
        
        losses = []
        pbar = tqdm(range(epochs), desc="Stage 2")
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Forward pass - Stage 2 style
            batch_size = points.size(0)
            num_points = points.size(1)
            
            # Variational encoding
            z_mu, z_sigma = self.model.encoder(points)
            
            # Reparameterization trick
            eps = torch.randn_like(z_mu)
            z = z_mu + z_sigma * eps
            
            # Entropy of encoder distribution
            entropy = gaussian_entropy(z_sigma)
            
            # Latent CNF for prior
            if self.model.use_latent_flow and self.model.latent_cnf is not None:
                w, delta_log_pw = self.model.latent_cnf(
                    z, None, torch.zeros(batch_size, 1).to(z)
                )
                log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
                delta_log_pw = delta_log_pw.view(batch_size, 1)
                log_pz = log_pw - delta_log_pw
            else:
                log_pz = standard_normal_logprob(z).view(batch_size, -1).sum(1, keepdim=True)
            
            # Point CNF for reconstruction
            z_new = z + (log_pz * 0.).mean()  # Gradient trick
            y, delta_log_py = self.model.point_cnf(
                points, z_new, torch.zeros(batch_size, num_points, 1).to(points)
            )
            
            log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
            log_px = log_py - delta_log_py
            
            # Full ELBO loss
            recon_loss = -log_px.mean()
            prior_loss = -log_pz.mean()
            entropy_loss = -entropy.mean()
            
            # Gradual KL annealing
            kl_weight = min(1.0, epoch / 100.0)
            loss = recon_loss + kl_weight * (prior_loss + entropy_loss)
            
            # Check for numerical issues
            if torch.isnan(loss) or loss.item() > 1e6:
                logger.warning(f"Numerical issue at epoch {epoch}: loss={loss.item()}")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{(prior_loss + entropy_loss).item():.4f}",
                "kl_w": f"{kl_weight:.3f}"
            })
            
            # Visualize periodically
            if (epoch + 1) % 50 == 0:
                self.visualize_reconstruction(points, z, epoch, stage=2)
        
        return losses
    
    def visualize_reconstruction(self, points, z, epoch, stage):
        """Visualize reconstruction quality."""
        with torch.no_grad():
            # Decode
            recon = self.model.decode(z, points.shape[1])
            
            # Save visualization
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Original
            orig = points[0].cpu().numpy()
            ax1.scatter(orig[:, 0], orig[:, 1], s=1, alpha=0.5)
            ax1.set_title(f"Original")
            ax1.set_aspect('equal')
            
            # Reconstruction
            rec = recon[0].cpu().numpy()
            ax2.scatter(rec[:, 0], rec[:, 1], s=1, alpha=0.5)
            ax2.set_title(f"Reconstruction (Stage {stage}, Epoch {epoch})")
            ax2.set_aspect('equal')
            
            plt.tight_layout()
            save_path = f"outputs/two_stage/stage{stage}_epoch{epoch:04d}.png"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory with slice data")
    parser.add_argument("--slice-name", help="Specific slice to train on")
    parser.add_argument("--stage1-epochs", type=int, default=500)
    parser.add_argument("--stage2-epochs", type=int, default=500)
    parser.add_argument("--lr-stage1", type=float, default=5e-4)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    points, slice_id = load_single_slice(args.data_dir, args.slice_name)
    points = points.to(args.device)
    
    # Normalize
    points_mean = points.mean(dim=1, keepdim=True)
    points_std = points.std(dim=1, keepdim=True).clamp(min=1e-6)
    points = (points - points_mean) / points_std
    
    logger.info(f"Normalized points: mean={points_mean.squeeze()}, std={points_std.squeeze()}")
    
    # Create model
    model = PointFlow2DVAE(
        input_dim=2,
        latent_dim=args.latent_dim,
        use_latent_flow=True,  # We'll use it in stage 2
        cnf_atol=1e-3,
        cnf_rtol=1e-3
    )
    
    # Create trainer
    trainer = TwoStageTrainer(model, args.device)
    
    # Stage 1: Autoencoder
    optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr_stage1)
    losses1 = trainer.train_stage1(points, optimizer1, args.stage1_epochs)
    
    # Save Stage 1 checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'stage': 1,
        'losses': losses1
    }, "outputs/two_stage/stage1_checkpoint.pt")
    
    # Stage 2: Full model (lower learning rate)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr_stage2)
    losses2 = trainer.train_stage2(points, optimizer2, args.stage2_epochs)
    
    # Save Stage 2 checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'stage': 2,
        'losses': losses2
    }, "outputs/two_stage/stage2_checkpoint.pt")
    
    logger.info("Two-stage training complete!")
    
    # Plot losses
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(losses1)
    ax1.set_title("Stage 1: Autoencoder Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Loss")
    
    ax2.plot(losses2)
    ax2.set_title("Stage 2: Full VAE Loss")
    ax2.set_xlabel("Epoch") 
    ax2.set_ylabel("ELBO Loss")
    
    plt.tight_layout()
    plt.savefig("outputs/two_stage/training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
