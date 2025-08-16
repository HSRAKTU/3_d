#!/usr/bin/env python3
"""
Simple multi-slice training using EXACT approach from successful single-slice overfitting.
No complex VAE losses, just the working reconstruction approach.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pointflow2d_adapted import PointFlow2DAdaptedVAE
from src.training.dataset import SliceDataset, collate_variable_length
from src.data.loader import SliceDataLoader

# Configuration from successful single-slice
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
MIN_LR = 1e-4
EPOCHS = 300
GRADIENT_CLIP = 1.0
WEIGHT_DECAY = 1e-4
LATENT_DIM = 128
SOLVER_STEPS = 5  # From Experiment B2

def compute_chamfer_distance(x, y):
    """Compute Chamfer distance between two point sets"""
    x_expanded = x.unsqueeze(1)
    y_expanded = y.unsqueeze(0)
    distances = torch.norm(x_expanded - y_expanded, dim=2)
    min_dist_x_to_y = distances.min(dim=1)[0]
    min_dist_y_to_x = distances.min(dim=0)[0]
    chamfer = min_dist_x_to_y.mean() + min_dist_y_to_x.mean()
    return chamfer.item()

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path('outputs/train_10cars_simple')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data for 10 cars...")
    data_loader = SliceDataLoader('data/training_dataset')
    car_ids = data_loader.get_car_ids()[:10]
    
    dataset = SliceDataset(
        data_directory='data/training_dataset',
        car_ids=car_ids,
        normalize=True,
        max_points=None,
        min_points=10
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_variable_length
    )
    
    print(f"Total slices: {len(dataset)}")
    
    # Create model - EXACT same as single-slice
    model = PointFlow2DAdaptedVAE(
        input_dim=2,
        latent_dim=LATENT_DIM,
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,
        solver='euler',
        solver_steps=SOLVER_STEPS,
        use_deterministic_encoder=True
    ).to(device)
    
    # Optimizer - EXACT same as single-slice
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    
    # Training loop - EXACT same approach as single-slice
    print("\nStarting training...")
    best_chamfer = float('inf')
    training_log = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_chamfer = 0
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            points = batch['points'].to(device)
            mask = batch['mask'].to(device)
            points = points * mask.unsqueeze(-1)
            
            # EXACT same training as single-slice
            # Enable gradients for divergence computation in CNF
            points.requires_grad_(True)
            
            z = model.encode(points)
            
            # Forward through CNF (real slice → Gaussian)
            y, log_det = model.point_cnf(points, z, reverse=False)
            
            # Negative log-likelihood under Gaussian
            log_py = torch.distributions.Normal(0, 1).log_prob(y)
            log_py = log_py.view(points.shape[0], -1).sum(1, keepdim=True)
            log_px = log_py + log_det.view(points.shape[0], -1).sum(1, keepdim=True)
            
            # Loss
            loss = -log_px.mean()
            
            # L2 regularization on output
            output_reg = 0.01 * (y ** 2).mean()
            loss = loss + output_reg
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()
            
            # Monitor Chamfer
            with torch.no_grad():
                x_recon = model.reconstruct(points)
                chamfer = compute_chamfer_distance(x_recon, points)
                total_chamfer += chamfer
            
            total_loss += loss.item()
            n_batches += 1
            
            # Clear cache
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Epoch stats
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_chamfer = total_chamfer / n_batches if n_batches > 0 else 0
        
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:7.4f} | Chamfer: {avg_chamfer:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save if better
        if avg_chamfer < best_chamfer:
            best_chamfer = avg_chamfer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_chamfer': best_chamfer
            }, output_dir / 'best_model.pth')
        
        # Log
        training_log.append({
            'epoch': epoch,
            'loss': avg_loss,
            'chamfer': avg_chamfer,
            'lr': scheduler.get_last_lr()[0]
        })
        
        with open(output_dir / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)
        
        scheduler.step()
    
    print(f"\nTraining complete! Best Chamfer: {best_chamfer:.4f}")

if __name__ == '__main__':
    main()
