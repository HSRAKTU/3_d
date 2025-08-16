#!/usr/bin/env python3
"""
Train PointFlow2D VAE with two-stage training using the exact architecture that worked.
Based on successful single-slice configuration.
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

from src.models.pointflow2d_adapted import PointFlow2DAdaptedVAE
from src.training.dataset import SliceDataset, collate_variable_length
from src.data.loader import SliceDataLoader

# Set environment for memory efficiency
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


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
    return chamfer.item()


def train_epoch(model, dataloader, optimizer, device, epoch, warmup_epochs):
    """Train for one epoch with two-stage training"""
    model.train()
    
    total_loss = 0
    total_chamfer = 0
    n_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        points = batch['points'].to(device)
        mask = batch['mask'].to(device)
        
        # Apply mask
        points = points * mask.unsqueeze(-1)
        
        optimizer.zero_grad()
        
        try:
            # Encode to latent
            z = model.encode(points)
            
            # Forward pass through CNF
            y, log_det = model.point_cnf(points, z, reverse=False)
            
            # Compute negative log-likelihood
            log_py = torch.distributions.Normal(0, 1).log_prob(y)
            log_py = log_py.view(points.shape[0], -1).sum(1, keepdim=True)
            log_px = log_py + log_det.view(points.shape[0], -1).sum(1, keepdim=True)
            
            # Primary loss
            nll_loss = -log_px.mean()
            
            # Add L2 regularization on CNF output
            output_reg = 0.01 * (y ** 2).mean()
            
            # Two-stage training
            if epoch < warmup_epochs:
                # Stage 1: Only reconstruction loss
                loss = nll_loss + output_reg
            else:
                # Stage 2: Add VAE regularization (simplified for now)
                # TODO: Add proper prior and entropy losses when implementing stochastic encoder
                loss = nll_loss + output_reg
            
            # Compute Chamfer for monitoring
            with torch.no_grad():
                x_recon = model.decode(z, points.shape[1])
                chamfer = compute_chamfer_distance(x_recon, points)
                total_chamfer += chamfer
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
            
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / n_batches, total_chamfer / n_batches


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
                # Reconstruct
                x_recon = model.reconstruct(points)
                
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
    
    # Model parameters (from successful single-slice)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--solver_steps', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_cars', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='outputs/vae_stable_v2')
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
        max_points=None,
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
    
    # Create model - use the exact working architecture
    model = PointFlow2DAdaptedVAE(
        point_dim=2,
        latent_dim=args.latent_dim,
        encoder_hidden_dim=args.hidden_dim,
        cnf_hidden_dim=args.hidden_dim,
        solver='euler',
        solver_steps=args.solver_steps,
        use_deterministic_encoder=True  # Always deterministic for now
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
    print(f"Stage 2 (Full VAE - TODO): epochs {args.warmup_epochs}-{args.epochs}\n")
    
    for epoch in range(args.epochs):
        # Train
        avg_loss, avg_chamfer = train_epoch(
            model, train_loader, optimizer, device, epoch, args.warmup_epochs
        )
        
        # Validate
        val_chamfer = validate(model, val_loader, device)
        
        # Log
        log_entry = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'train_chamfer': avg_chamfer,
            'val_chamfer': val_chamfer,
            'lr': scheduler.get_last_lr()[0],
            'stage': 1 if epoch < args.warmup_epochs else 2
        }
        training_log.append(log_entry)
        
        # Print progress
        stage = "Stage 1" if epoch < args.warmup_epochs else "Stage 2"
        print(f"{stage} - Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {avg_loss:7.4f} | Train Chamfer: {avg_chamfer:.4f} | Val Chamfer: {val_chamfer:.4f}")
        
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


if __name__ == '__main__':
    main()
