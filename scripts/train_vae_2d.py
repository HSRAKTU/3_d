#!/usr/bin/env python3
"""
Train PointFlow2D VAE with proper three-loss formulation.
Robust CLI script designed for POD deployment.
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.encoder import PointNet2DEncoder
from src.models.pointflow2d_adapted import PointFlow2DODE
from src.training.dataset import SliceDataset
from src.data.loader import SliceDataLoader

# Set environment for memory efficiency
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class PointFlow2DVAE(nn.Module):
    """2D PointFlow VAE with proper ELBO loss"""
    
    def __init__(
        self,
        point_dim=2,
        latent_dim=128,
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,
        solver='euler',
        solver_steps=10,
        use_stochastic=True,
        lambda_recon=1.0,
        lambda_prior=0.1,
        lambda_entropy=0.01,
        lambda_chamfer=10.0,
        lambda_volume=0.01
    ):
        super().__init__()
        
        # Architecture
        self.latent_dim = latent_dim
        self.use_stochastic = use_stochastic
        
        # Encoder always outputs mu and logvar
        self.encoder = PointNet2DEncoder(
            input_dim=point_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=latent_dim
        )
        
        # Point CNF (decoder)
        self.cnf_func = PointFlow2DODE(
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
        self.lambda_volume = lambda_volume
        
    def encode(self, x):
        """Encode points to latent distribution"""
        # Encoder always outputs mu and logvar
        mu, logvar = self.encoder(x)
        
        if self.use_stochastic:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            return z, mu, logvar
        else:
            # Deterministic encoding - just use mu
            return mu, None, None
            
    def cnf_forward(self, x, z):
        """Forward pass through CNF: data -> Gaussian"""
        from torchdiffeq import odeint
        
        B, N, D = x.shape
        
        # Ensure gradients for divergence computation
        x = x.requires_grad_(True)
        
        # Expand context for all points
        z_expanded = z.unsqueeze(1).expand(B, N, -1)
        
        # Initial state
        states = (x, z_expanded)
        
        # Integrate forward (data to Gaussian)
        if self.solver == 'euler':
            # Fixed step Euler
            dt = 1.0 / self.solver_steps
            log_det = torch.zeros(B, N, 1).to(x)
            
            for i in range(self.solver_steps):
                t = torch.tensor(i * dt).to(x)
                v, div = self.cnf_func(t, states)
                
                states = (states[0] + dt * v, states[1])
                log_det = log_det + dt * div
                
            y = states[0]
        else:
            # Adaptive solver
            t = torch.tensor([0., 1.]).to(x)
            sol = odeint(
                self.cnf_func,
                states,
                t,
                method=self.solver,
                atol=self.atol,
                rtol=self.rtol
            )
            y = sol[0][-1]
            log_det = sol[1][-1] if len(sol) > 1 else torch.zeros(B, N, 1).to(x)
            
        return y, log_det
        
    def sample(self, z, num_points):
        """Sample points from latent code (generation)"""
        from torchdiffeq import odeint
        
        B = z.shape[0]
        device = z.device
        
        # Sample from Gaussian
        y = torch.randn(B, num_points, 2).to(device)
        
        # Expand context
        z_expanded = z.unsqueeze(1).expand(B, num_points, -1)
        
        # Reverse integration (Gaussian to data)
        states = (y, z_expanded)
        
        if self.solver == 'euler':
            dt = 1.0 / self.solver_steps
            for i in range(self.solver_steps):
                t = torch.tensor(1.0 - i * dt).to(device)
                with torch.no_grad():  # No gradients needed for sampling
                    # Get velocity only, ignore divergence
                    inputs = torch.cat([states[0], 
                                      torch.ones(B, num_points, 1).to(device) * t.view(1, 1, 1),
                                      states[1]], dim=-1)
                    v = self.cnf_func.net(inputs)
                states = (states[0] - dt * v, states[1])
            x = states[0]
        else:
            t = torch.tensor([1., 0.]).to(device)
            sol = odeint(
                lambda t, states: (-self.cnf_func(1-t, states)[0], states[1]),
                states,
                t,
                method=self.solver,
                atol=self.atol,
                rtol=self.rtol
            )
            x = sol[0][-1]
            
        return x
        
    def compute_losses(self, x):
        """Compute all VAE losses"""
        B, N, D = x.shape
        
        # 1. Encode
        z, mu, logvar = self.encode(x)
        
        # 2. Forward through CNF
        y, log_det = self.cnf_forward(x, z)
        
        # Loss 1: Reconstruction Loss
        # y should be N(0,I)
        log_py = -0.5 * (y ** 2 + np.log(2 * np.pi)).sum(dim=-1)  # [B, N]
        log_px = log_py + log_det.squeeze(-1)  # [B, N]
        recon_loss = -log_px.mean()
        
        # Loss 2: Prior Loss (KL divergence)
        if self.use_stochastic:
            # KL(q(z|x) || p(z)) where p(z) = N(0,I)
            prior_loss = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(dim=1).mean()
        else:
            # L2 regularization for deterministic
            prior_loss = 0.5 * (z ** 2).sum(dim=1).mean()
            
        # Loss 3: Entropy Loss
        if self.use_stochastic:
            # Maximize entropy: H(q(z|x)) = 0.5 * sum(log(2πe) + logvar)
            entropy = 0.5 * (logvar + 1 + np.log(2 * np.pi)).sum(dim=1)
            entropy_loss = -entropy.mean()  # Negative because we maximize
        else:
            entropy_loss = torch.tensor(0.0).to(x)
            
        # Loss 4: Chamfer Distance (auxiliary)
        with torch.no_grad():
            x_recon = self.sample(z, N)
        chamfer_loss = compute_chamfer_distance(x_recon, x)
        
        # Loss 5: Volume Regularization (prevent explosion)
        volume_reg = torch.relu(log_det.mean() - 10.0)
        
        # Total loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_prior * prior_loss +
            self.lambda_entropy * entropy_loss +
            self.lambda_chamfer * chamfer_loss +
            self.lambda_volume * volume_reg
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'prior': prior_loss.item(),
            'entropy': entropy_loss.item() if self.use_stochastic else 0.0,
            'chamfer': chamfer_loss.item(),
            'volume': volume_reg.item(),
            'log_det_mean': log_det.mean().item()
        }


def compute_chamfer_distance(x, y):
    """Compute Chamfer distance between two point sets"""
    x_expanded = x.unsqueeze(2)  # [B, N, 1, D]
    y_expanded = y.unsqueeze(1)  # [B, 1, M, D]
    
    distances = torch.norm(x_expanded - y_expanded, dim=3)  # [B, N, M]
    
    min_dist_x_to_y = distances.min(dim=2)[0]  # [B, N]
    min_dist_y_to_x = distances.min(dim=1)[0]  # [B, M]
    
    chamfer = min_dist_x_to_y.mean() + min_dist_y_to_x.mean()
    return chamfer


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {
        'recon': 0, 'prior': 0, 'entropy': 0, 
        'chamfer': 0, 'volume': 0, 'log_det_mean': 0
    }
    
    for batch_idx, (points, mask, _) in enumerate(dataloader):
        points = points.to(device)
        mask = mask.to(device)
        
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
                if k != 'total':
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
        for points, mask, _ in dataloader:
            points = points.to(device)
            mask = mask.to(device)
            points = points * mask.unsqueeze(-1)
            
            try:
                # Encode and reconstruct
                z, _, _ = model.encode(points)
                x_recon = model.sample(z, points.shape[1])
                
                # Compute Chamfer
                chamfer = compute_chamfer_distance(x_recon, points)
                total_chamfer += chamfer.item()
                n_batches += 1
                
            except Exception as e:
                print(f"Validation error: {e}")
                continue
                
    return total_chamfer / n_batches if n_batches > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description='Train PointFlow2D VAE')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--solver_steps', type=int, default=10)
    parser.add_argument('--use_stochastic', action='store_true', default=False)
    
    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_prior', type=float, default=0.1)
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--lambda_chamfer', type=float, default=10.0)
    parser.add_argument('--lambda_volume', type=float, default=0.01)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_cars', type=int, default=10)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/vae_training')
    parser.add_argument('--save_freq', type=int, default=50)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    print(f"Loading data for {args.num_cars} cars...")
    loader = SliceDataLoader()
    car_ids = loader.get_car_ids()[:args.num_cars]
    
    train_dataset = SliceDataset(
        data_loader=loader,
        car_ids=car_ids,
        max_points=None,  # Use all points
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Split for validation
    val_size = len(train_dataset) // 10
    val_dataset = torch.utils.data.Subset(train_dataset, range(val_size))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Training samples: {len(train_dataset) - val_size}")
    print(f"Validation samples: {val_size}")
    
    # Create model
    model = PointFlow2DVAE(
        latent_dim=args.latent_dim,
        cnf_hidden_dim=args.hidden_dim,
        solver_steps=args.solver_steps,
        use_stochastic=args.use_stochastic,
        lambda_recon=args.lambda_recon,
        lambda_prior=args.lambda_prior,
        lambda_entropy=args.lambda_entropy,
        lambda_chamfer=args.lambda_chamfer,
        lambda_volume=args.lambda_volume
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Training log
    log_file = output_dir / 'training_log.json'
    training_log = []
    
    # Training loop
    best_chamfer = float('inf')
    
    print("\nStarting training...")
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
            **loss_components
        }
        training_log.append(log_entry)
        
        # Print progress
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f} - "
              f"Chamfer: {val_chamfer:.4f} - "
              f"Recon: {loss_components['recon']:.3f} - "
              f"Prior: {loss_components['prior']:.3f} - "
              f"LogDet: {loss_components['log_det_mean']:.3f}")
        
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
            
        # Save log
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
            
        scheduler.step()
        
    print(f"\nTraining complete! Best Chamfer: {best_chamfer:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
