#!/usr/bin/env python3
"""
Train PointFlow2D VAE with two-stage training and stability fixes.
Based on successful single-slice configuration and original PointFlow approach.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.encoder import PointNet2DEncoder
from src.models.pointflow2d_adapted import PointFlow2DODE
from src.training.dataset import SliceDataset, collate_variable_length
from src.data.loader import SliceDataLoader

# Set environment for memory efficiency
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class StablePointFlow2DODE(nn.Module):
    """2D PointFlow ODE with bounded transformations for stability"""
    
    def __init__(self, point_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.point_dim = point_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Use bounded activations for stability
        self.net = nn.Sequential(
            nn.Linear(point_dim + 1 + context_dim, hidden_dim),
            nn.Tanh(),  # Bounded [-1, 1]
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, point_dim)
        )
        
        # Output scaling to control transformation magnitude
        self.output_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, t, states):
        x = states[0]  # Points [B, N, D]
        context = states[1]  # Latent context [B, N, latent_dim]
        
        B, N, D = x.shape
        
        # Time embedding
        t_vec = torch.ones(B, N, 1).to(x) * t.view(1, 1, 1)
        
        # Concatenate inputs
        inputs = torch.cat([x, t_vec, context], dim=-1)
        
        # Get velocity field (bounded by tanh)
        v = self.net(inputs) * self.output_scale
        
        # Compute divergence for log determinant
        if self.training:
            # Enable gradients for divergence computation
            if not x.requires_grad:
                x.requires_grad_(True)
                
            divergence = 0.0
            for i in range(self.point_dim):
                divergence += torch.autograd.grad(
                    v[:, :, i].sum(), x,
                    create_graph=True, retain_graph=True
                )[0][:, :, i]
                
            divergence = divergence.unsqueeze(-1)
        else:
            divergence = torch.zeros(B, N, 1).to(x)
        
        # Pad to match context dimension
        v_padded = torch.zeros(B, N, context.shape[-1]).to(v)
        v_padded[:, :, :self.point_dim] = v
        
        return v_padded, divergence


class StablePointFlow2DVAE(nn.Module):
    """2D PointFlow VAE with two-stage training and stability improvements"""
    
    def __init__(
        self,
        point_dim=2,
        latent_dim=128,
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,
        solver='euler',
        solver_steps=5,  # From successful single-slice
        warmup_epochs=100,  # Two-stage training
        lambda_recon=1.0,
        lambda_prior=0.1,
        lambda_entropy=0.01,
        lambda_chamfer=10.0,
        lambda_logdet=1.0  # Strong log det regularization
    ):
        super().__init__()
        
        # Architecture
        self.latent_dim = latent_dim
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Encoder (always outputs mu and logvar, but we use deterministic initially)
        self.encoder = PointNet2DEncoder(
            input_dim=point_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=latent_dim
        )
        
        # Stable Point CNF
        self.cnf_func = StablePointFlow2DODE(
            point_dim=point_dim,
            context_dim=latent_dim,
            hidden_dim=cnf_hidden_dim
        )
        
        # ODE solver settings
        self.solver = solver
        self.solver_steps = solver_steps
        self.atol = 1e-3
        self.rtol = 1e-3
        
        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_prior = lambda_prior
        self.lambda_entropy = lambda_entropy
        self.lambda_chamfer = lambda_chamfer
        self.lambda_logdet = lambda_logdet
        
    def encode(self, x):
        """Encode points to latent distribution"""
        mu, logvar = self.encoder(x)
        
        # Two-stage: deterministic during warmup, stochastic after
        if self.current_epoch < self.warmup_epochs:
            return mu, None, None
        else:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
            
    def cnf_forward(self, x, z):
        """Forward pass through CNF (real points -> Gaussian)"""
        B, N, D = x.shape
        
        # Expand latent code to all points
        z_expanded = z.unsqueeze(1).expand(B, N, self.latent_dim)
        
        # Set up ODE states
        states = (x, z_expanded)
        
        # Integrate forward (t=0 to t=1)
        if self.solver == 'euler':
            dt = 1.0 / self.solver_steps
            for i in range(self.solver_steps):
                t = torch.tensor(i * dt).to(x)
                derivs = self.cnf_func(t, states)
                states = tuple(s + d * dt for s, d in zip(states, derivs))
        else:
            raise NotImplementedError(f"Solver {self.solver} not implemented")
            
        y = states[0][:, :, :D]  # Extract points
        
        # Accumulate log determinant
        log_det = torch.zeros(B, N, 1).to(x)
        dt = 1.0 / self.solver_steps
        states_temp = (x, z_expanded)
        
        for i in range(self.solver_steps):
            t = torch.tensor(i * dt).to(x)
            _, div = self.cnf_func(t, states_temp)
            log_det = log_det + div * dt
            
            # Update states for next step
            derivs = self.cnf_func(t, states_temp)
            states_temp = tuple(s + d[:, :, :s.shape[-1]] * dt 
                              for s, d in zip(states_temp, derivs))
        
        return y, log_det
        
    def sample(self, z, num_points):
        """Sample points from latent code (Gaussian -> real points)"""
        B = z.shape[0]
        device = z.device
        
        # Sample from standard normal
        y = torch.randn(B, num_points, 2).to(device)
        
        # Expand latent code
        z_expanded = z.unsqueeze(1).expand(B, num_points, self.latent_dim)
        
        # Set up ODE states
        states = (y, z_expanded)
        
        # Integrate backward (t=1 to t=0)
        if self.solver == 'euler':
            dt = 1.0 / self.solver_steps
            for i in range(self.solver_steps):
                t = torch.tensor(1.0 - i * dt).to(y)
                derivs = self.cnf_func(t, states)
                states = tuple(s - d[:, :, :s.shape[-1]] * dt 
                             for s, d in zip(states, derivs))
        
        x = states[0]
        return x
        
    def compute_losses(self, x):
        """Compute VAE losses with two-stage training"""
        B, N, D = x.shape
        device = x.device
        
        # 1. Encode to latent
        z, mu, logvar = self.encode(x)
        
        # 2. Transform points to Gaussian
        y, log_det = self.cnf_forward(x, z)
        
        # 3. Compute reconstruction loss (always active)
        log_py = torch.distributions.Normal(0, 1).log_prob(y).view(B, -1).sum(1)
        log_px = log_py + log_det.view(B, -1).sum(1)
        recon_loss = -log_px.mean()
        
        # 4. Strong log det regularization (always active)
        # Penalize both positive and negative deviations from 0
        logdet_penalty = (log_det**2).mean() + (log_det.abs()).mean()
        
        # 5. Chamfer loss (always active for monitoring)
        with torch.no_grad():
            x_recon = self.sample(z, N)
        chamfer = compute_chamfer_distance(x_recon, x)
        
        # Two-stage training
        if self.current_epoch < self.warmup_epochs:
            # Stage 1: Only reconstruction + regularization
            total_loss = (self.lambda_recon * recon_loss + 
                         self.lambda_logdet * logdet_penalty +
                         self.lambda_chamfer * chamfer)
            
            # Zero out VAE losses during warmup
            prior_loss = torch.tensor(0.0).to(device)
            entropy_loss = torch.tensor(0.0).to(device)
        else:
            # Stage 2: Full VAE losses
            if logvar is not None:
                # Prior loss (KL divergence)
                prior_loss = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(dim=1).mean()
                
                # Entropy loss
                entropy = 0.5 * (logvar + 1 + np.log(2 * np.pi)).sum(dim=1)
                entropy_loss = -entropy.mean()
            else:
                prior_loss = torch.tensor(0.0).to(device)
                entropy_loss = torch.tensor(0.0).to(device)
                
            total_loss = (self.lambda_recon * recon_loss + 
                         self.lambda_prior * prior_loss + 
                         self.lambda_entropy * entropy_loss +
                         self.lambda_logdet * logdet_penalty +
                         self.lambda_chamfer * chamfer)
        
        return total_loss, {
            'recon': recon_loss.item(),
            'prior': prior_loss.item() if isinstance(prior_loss, torch.Tensor) else prior_loss,
            'entropy': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
            'chamfer': chamfer,
            'logdet_penalty': logdet_penalty.item(),
            'log_det_mean': log_det.mean().item(),
            'log_det_std': log_det.std().item()
        }
        
    def set_epoch(self, epoch):
        """Update current epoch for two-stage training"""
        self.current_epoch = epoch


def compute_chamfer_distance(x, y):
    """Compute Chamfer distance between two point sets"""
    # x: [B, N, D], y: [B, M, D]
    x_expanded = x.unsqueeze(2)  # [B, N, 1, D]
    y_expanded = y.unsqueeze(1)  # [B, 1, M, D]
    
    # Compute pairwise distances
    distances = torch.norm(x_expanded - y_expanded, dim=3)  # [B, N, M]
    
    # Chamfer distance
    min_dist_x_to_y = distances.min(dim=2)[0]  # [B, N]
    min_dist_y_to_x = distances.min(dim=1)[0]  # [B, M]
    
    chamfer = min_dist_x_to_y.mean() + min_dist_y_to_x.mean()
    return chamfer


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    model.set_epoch(epoch)  # Update epoch for two-stage training
    
    total_loss = 0
    loss_components = {
        'recon': 0, 'prior': 0, 'entropy': 0, 
        'chamfer': 0, 'logdet_penalty': 0, 'log_det_mean': 0, 'log_det_std': 0
    }
    
    for batch_idx, batch in enumerate(dataloader):
        points = batch['points'].to(device)
        mask = batch['mask'].to(device)
        
        # Apply mask
        points = points * mask.unsqueeze(-1)
        
        optimizer.zero_grad()
        
        try:
            loss, losses = model.compute_losses(points)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}, skipping")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
            
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v/n_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def validate(model, dataloader, device):
    """Validation"""
    model.eval()
    total_chamfer = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            points = batch['points'].to(device)
            mask = batch['mask'].to(device)
            points = points * mask.unsqueeze(-1)
            
            try:
                # Encode and reconstruct
                z, _, _ = model.encode(points)
                x_recon = model.sample(z, points.shape[1])
                
                # Compute Chamfer distance
                chamfer = compute_chamfer_distance(x_recon, points)
                total_chamfer += chamfer
                n_batches += 1
                
            except Exception as e:
                print(f"Validation error: {e}")
                continue
                
    return total_chamfer / n_batches if n_batches > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description='Train stable PointFlow2D VAE')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--solver_steps', type=int, default=5)  # From successful single-slice
    parser.add_argument('--warmup_epochs', type=int, default=100)  # Two-stage training
    
    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_prior', type=float, default=0.1)
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--lambda_chamfer', type=float, default=10.0)
    parser.add_argument('--lambda_logdet', type=float, default=1.0)  # Strong regularization
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=3e-4)  # From successful single-slice
    parser.add_argument('--num_cars', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='outputs/vae_stable')
    parser.add_argument('--save_freq', type=int, default=50)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    print(f"Loading data for {args.num_cars} cars...")
    data_loader = SliceDataLoader('data/training_dataset')
    car_ids = data_loader.get_car_ids()[:args.num_cars]
    
    train_dataset = SliceDataset(
        data_directory='data/training_dataset',
        car_ids=car_ids,
        normalize=True,
        max_points=None,  # Use all points
        min_points=10
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_variable_length
    )
    
    # Split for validation
    val_size = len(train_dataset) // 10
    val_dataset = torch.utils.data.Subset(train_dataset, range(val_size))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length
    )
    
    print(f"Training samples: {len(train_dataset) - val_size}")
    print(f"Validation samples: {val_size}")
    
    # Create model
    model = StablePointFlow2DVAE(
        latent_dim=args.latent_dim,
        encoder_hidden_dim=args.hidden_dim,
        cnf_hidden_dim=args.hidden_dim,
        solver_steps=args.solver_steps,
        warmup_epochs=args.warmup_epochs,
        lambda_recon=args.lambda_recon,
        lambda_prior=args.lambda_prior,
        lambda_entropy=args.lambda_entropy,
        lambda_chamfer=args.lambda_chamfer,
        lambda_logdet=args.lambda_logdet
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Training log
    log_file = output_dir / 'training_log.json'
    training_log = []
    
    # Training loop
    best_chamfer = float('inf')
    
    print("\nStarting two-stage training...")
    print(f"Stage 1 (Reconstruction only): epochs 0-{args.warmup_epochs}")
    print(f"Stage 2 (Full VAE): epochs {args.warmup_epochs}-{args.epochs}\n")
    
    for epoch in range(args.epochs):
        # Train
        avg_loss, loss_components = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Validate
        val_chamfer = validate(model, val_loader, device)
        
        # Log
        log_entry = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'val_chamfer': val_chamfer,
            'lr': scheduler.get_last_lr()[0],
            'stage': 1 if epoch < args.warmup_epochs else 2,
            **loss_components
        }
        training_log.append(log_entry)
        
        # Print progress
        stage = "Stage 1" if epoch < args.warmup_epochs else "Stage 2"
        print(f"{stage} - Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:7.4f} | Chamfer: {val_chamfer:.4f} | "
              f"Recon: {loss_components['recon']:7.3f} | Prior: {loss_components['prior']:6.3f} | "
              f"LogDet: {loss_components['log_det_mean']:6.3f} (±{loss_components['log_det_std']:.3f}) | "
              f"LogDet Penalty: {loss_components['logdet_penalty']:6.3f}")
        
        # Stability check
        if loss_components['log_det_mean'] > 3.0 or loss_components['recon'] < -1.0:
            print(f"\n⚠️  WARNING: Training becoming unstable! Consider stopping and adjusting hyperparameters.")
        
        # Save best model
        if val_chamfer < best_chamfer:
            best_chamfer = val_chamfer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_chamfer': val_chamfer,
                'config': vars(args)
            }, output_dir / 'best_model.pth')
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_chamfer': val_chamfer,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
            
        # Save log after each epoch
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
            
        scheduler.step()
        
    print(f"\nTraining complete! Best Chamfer: {best_chamfer:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Final summary
    with open(output_dir / 'final_summary.txt', 'w') as f:
        f.write(f"Training completed\n")
        f.write(f"Best validation Chamfer: {best_chamfer:.4f}\n")
        f.write(f"Final epoch: {epoch}\n")
        f.write(f"Final losses:\n")
        for k, v in loss_components.items():
            f.write(f"  {k}: {v:.4f}\n")


if __name__ == '__main__':
    main()
