#!/usr/bin/env python3
"""
Enhanced PointFlow2D training with comprehensive validation and monitoring.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import time
import json
import random
from pathlib import Path
from tqdm import tqdm
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_fixed import PointFlow2DVAE_Fixed
from training.dataset import SliceDataset, collate_variable_length
from utils.validation import TrainingMonitor, setup_validation_slice
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
    return worker_init_fn


def setup_logging():
    """Setup comprehensive logging."""
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "enhanced_training.log")
        ]
    )
    return logs_dir


def get_beta_schedule(epoch: int, epochs: int, schedule: str = "linear", 
                     start: float = 0.0, end: float = 1.0) -> float:
    """Get beta value for KL annealing."""
    if schedule == "constant":
        return end
    elif schedule == "linear":
        return start + (end - start) * (epoch / epochs)
    elif schedule == "cosine":
        return start + (end - start) * (1 - np.cos(np.pi * epoch / epochs)) / 2
    else:
        return end


def save_checkpoint(model, optimizer, epoch, loss_history, save_dir, metadata=None):
    """Save enhanced checkpoint with comprehensive metadata."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get model info
    model_info = model.get_model_info()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'model_config': {
            'latent_dim': model.latent_dim,
            'input_dim': model.input_dim,
            'use_latent_flow': model.use_latent_flow,
            'total_parameters': model_info['total_parameters'],
            'encoder_parameters': model_info['encoder_parameters'],
            'point_cnf_parameters': model_info['point_cnf_parameters'],
            'latent_cnf_parameters': model_info['latent_cnf_parameters']
        },
        'metadata': metadata or {},
        'timestamp': time.time(),
        'pytorch_version': torch.__version__
    }
    
    # Save epoch checkpoint
    torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch:04d}.pt")
    
    # Save latest checkpoint
    torch.save(checkpoint, save_path / "latest_checkpoint.pt")
    
    # Save training metadata as JSON
    metadata_dict = {
        'current_epoch': epoch,
        'model_architecture': checkpoint['model_config'],
        'loss_history': loss_history,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pytorch_version': torch.__version__
    }
    
    with open(save_path / "training_metadata.json", 'w') as f:
        json.dump(metadata_dict, f, indent=2)


def load_checkpoint(model, optimizer, save_dir):
    """Load latest checkpoint if available."""
    checkpoint_path = Path(save_dir) / "latest_checkpoint.pt"
    
    if not checkpoint_path.exists():
        return 0, []
    
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get epoch and history
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint.get('loss_history', [])
        
        logging.info(f"Resumed from epoch {checkpoint['epoch']}")
        return start_epoch, loss_history
        
    except Exception as e:
        logging.warning(f"Failed to load checkpoint: {e}")
        return 0, []


def train_enhanced(model, train_loader, validation_slice, device, epochs=50, lr=1e-3,
                  beta_schedule="linear", save_every=5, validation_frequency=5,
                  monitoring_frequency=10, auto_resume=True, save_dir="outputs/enhanced"):
    """Enhanced training with comprehensive validation."""
    
    # Setup paths
    outputs_dir = Path(__file__).parent.parent / "outputs"
    checkpoints_dir = outputs_dir / "checkpoints_enhanced" 
    validation_dir = outputs_dir / "validation_enhanced"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup training monitor
    monitor = TrainingMonitor(
        save_dir=str(validation_dir),
        model_name="pointflow2d_enhanced",
        save_frequency=monitoring_frequency,
        visualization_frequency=validation_frequency
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Auto-resume if requested
    start_epoch = 0
    loss_history = []
    
    if auto_resume:
        start_epoch, loss_history = load_checkpoint(model, optimizer, checkpoints_dir)
    
    logging.info(f"Model: {model.get_model_info()}")
    logging.info(f"Starting training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        
        # Beta annealing
        beta = get_beta_schedule(epoch, epochs, beta_schedule, start=0.0, end=1.0)
        
        # Training metrics
        total_loss_accum = 0.0
        recon_loss_accum = 0.0
        prior_loss_accum = 0.0
        kl_loss_accum = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            points = batch['points'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            loss_dict = model.compute_loss(points, beta=beta)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss_accum += loss_dict['total_loss'].item()
            recon_loss_accum += loss_dict['recon_loss'].item()
            prior_loss_accum += loss_dict['prior_loss'].item()
            kl_loss_accum += loss_dict['kl_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'Recon': f"{loss_dict['recon_loss'].item():.4f}",
                'Prior': f"{loss_dict['prior_loss'].item():.4f}",
                'KL': f"{loss_dict['kl_loss'].item():.4f}",
                'Beta': f"{beta:.4f}"
            })
        
        # Calculate average losses
        avg_losses = {
            'total_loss': total_loss_accum / num_batches if num_batches > 0 else 0.0,
            'recon_loss': recon_loss_accum / num_batches if num_batches > 0 else 0.0,
            'prior_loss': prior_loss_accum / num_batches if num_batches > 0 else 0.0,
            'kl_loss': kl_loss_accum / num_batches if num_batches > 0 else 0.0
        }
        
        # Log training metrics to monitor
        monitor.log_training_step(epoch, avg_losses, model, optimizer)
        
        # Update loss history
        loss_entry = {
            'epoch': epoch,
            'total_loss': avg_losses['total_loss'],
            'recon_loss': avg_losses['recon_loss'],
            'prior_loss': avg_losses['prior_loss'],
            'kl_loss': avg_losses['kl_loss'],
            'beta': float(beta),
            'lr': optimizer.param_groups[0]['lr']
        }
        loss_history.append(loss_entry)
        
        # Enhanced validation and monitoring
        if epoch % validation_frequency == 0 or epoch == 0:
            logging.info(f"Running validation at epoch {epoch}...")
            
            # Reconstruction validation
            recon_metrics = monitor.validate_reconstruction(model, validation_slice, epoch)
            logging.info(f"Reconstruction metrics: {recon_metrics}")
            
            # Sampling validation (less frequent)
            if epoch % monitoring_frequency == 0:
                sampling_metrics = monitor.validate_sampling(model, num_samples=3, epoch=epoch)
                logging.info(f"Sampling metrics: {sampling_metrics}")
        
        # Check for training issues
        if epoch % 5 == 0:  # Check every 5 epochs
            issues = monitor.detect_training_issues()
            if issues:
                for issue in issues:
                    logging.warning(f"Training issue detected: {issue}")
        
        # Save training plots and validation summaries
        if epoch % monitoring_frequency == 0:
            monitor.save_training_plots(epoch)
            monitor.save_validation_summary(epoch)
        
        # Save checkpoint
        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss_history, checkpoints_dir)
            logging.info(f"Checkpoint saved at epoch {epoch}")
        
        # Log epoch summary
        logging.info(f"Epoch {epoch}: Loss={avg_losses['total_loss']:.4f}, "
                    f"Recon={avg_losses['recon_loss']:.4f}, "
                    f"Prior={avg_losses['prior_loss']:.4f}, "
                    f"KL={avg_losses['kl_loss']:.4f}")
    
    # Final checkpoint
    save_checkpoint(model, optimizer, epochs-1, loss_history, checkpoints_dir)
    monitor.save_training_plots(epochs-1)
    monitor.save_validation_summary(epochs-1)
    
    logging.info("Training completed!")
    return loss_history


def main():
    parser = argparse.ArgumentParser(description="Enhanced PointFlow2D Training")
    parser.add_argument("data_dir", help="Directory containing .npy slice files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--encoder-hidden-dim", type=int, default=256, help="Encoder hidden dimension")
    parser.add_argument("--cnf-hidden-dim", type=int, default=128, help="Point CNF hidden dimension")
    parser.add_argument("--latent-cnf-hidden-dim", type=int, default=128, help="Latent CNF hidden dimension")
    parser.add_argument("--beta-schedule", type=str, default="linear", 
                       choices=["constant", "linear", "cosine"], help="Beta annealing schedule")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--validation-frequency", type=int, default=5, help="Validation frequency")
    parser.add_argument("--monitoring-frequency", type=int, default=10, help="Monitoring frequency")
    parser.add_argument("--validation-slice", type=str, default=None, help="Specific slice file for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--auto-resume", action="store_true", default=True, help="Auto-resume from latest checkpoint")
    
    args = parser.parse_args()
    
    # Setup logging
    logs_dir = setup_logging()
    
    # Set random seed
    worker_init_fn = set_seed(args.seed)
    
    logging.info("Starting enhanced PointFlow2D training with comprehensive validation...")
    logging.info(f"Arguments: {vars(args)}")
    
    # Setup validation slice
    try:
        validation_slice = setup_validation_slice(args.data_dir, args.validation_slice)
        logging.info(f"Validation slice shape: {validation_slice.shape}")
    except Exception as e:
        logging.error(f"Failed to setup validation slice: {e}")
        return
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    # Create model
    logging.info("Creating enhanced PointFlow2D model...")
    model = PointFlow2DVAE_Fixed(
        latent_dim=args.latent_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        cnf_hidden_dim=args.cnf_hidden_dim,
        latent_cnf_hidden_dim=args.latent_cnf_hidden_dim,
        use_latent_flow=True
    ).to(device)
    
    # Log model info
    model_info = model.get_model_info()
    logging.info(f"Model created with {model_info['total_parameters']:,} parameters")
    logging.info(f"  - Encoder: {model_info['encoder_parameters']:,}")
    logging.info(f"  - Point CNF: {model_info['point_cnf_parameters']:,}")
    logging.info(f"  - Latent CNF: {model_info['latent_cnf_parameters']:,}")
    logging.info(f"  - Uses Latent Flow: {model_info['use_latent_flow']}")
    
    # Create dataset and dataloader
    logging.info(f"Loading dataset from: {args.data_dir}")
    dataset = SliceDataset(args.data_dir)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    
    logging.info(f"Dataset loaded: {len(dataset)} slices")
    
    # Train model
    logging.info("Starting enhanced training...")
    final_history = train_enhanced(
        model=model,
        train_loader=train_loader,
        validation_slice=validation_slice,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        beta_schedule=args.beta_schedule,
        save_every=args.save_every,
        validation_frequency=args.validation_frequency,
        monitoring_frequency=args.monitoring_frequency,
        auto_resume=args.auto_resume
    )
    
    final_loss = final_history[-1]['total_loss'] if final_history else 0.0
    logging.info(f"Training completed! Final loss: {final_loss:.6f}")


if __name__ == "__main__":
    main()
