#!/usr/bin/env python3
"""
Full reconstruction training on 10 cars with PointFlow CNF.
Phase 1: Pure reconstruction test - no validation split.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_fixed import PointFlow2DVAE_Fixed
from training.dataset import SliceDataset, collate_variable_length
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
    # For DataLoader workers
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
            logging.FileHandler(logs_dir / 'full_reconstruction_training.log')
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, losses, save_dir, metadata=None):
    """Save comprehensive checkpoint with metadata."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get model info
    model_info = model.get_model_info()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'model_config': {
            'latent_dim': model.latent_dim,
            'input_dim': model.input_dim,
            'encoder_hidden_dim': getattr(model, 'encoder_hidden_dim', 128),
            'cnf_hidden_dim': getattr(model, 'cnf_hidden_dim', 64),
            'total_parameters': model_info['total_parameters'],
            'encoder_parameters': model_info['encoder_parameters'],
            'cnf_parameters': model_info['cnf_parameters']
        },
        'metadata': metadata or {},
        'timestamp': time.time(),
        'pytorch_version': torch.__version__
    }
    
    # Save epoch checkpoint
    torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch:03d}.pt")
    
    # Save latest checkpoint
    torch.save(checkpoint, save_path / "latest_checkpoint.pt")
    
    # Save training metadata as JSON
    training_metadata = {
        'training_progress': {
            'current_epoch': epoch,
            'losses': losses,
            'best_reconstruction': max(losses['recon_loss']) if losses['recon_loss'] else float('-inf'),
            'final_reconstruction': losses['recon_loss'][-1] if losses['recon_loss'] else float('-inf')
        },
        'model_architecture': checkpoint['model_config'],
        'training_metadata': checkpoint['metadata'],
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pytorch_version': torch.__version__
    }
    
    # Save detailed metadata
    with open(save_path / "training_metadata.json", 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    # Save losses separately for easy plotting
    with open(save_path / "training_losses.json", 'w') as f:
        json.dump(losses, f, indent=2)


def train_full_reconstruction(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 4,
    latent_dim: int = 32,
    cnf_hidden: int = 64,
    lr: float = 5e-5,
    beta_schedule: str = "linear",
    beta_start: float = 0.0,
    beta_end: float = 0.01,
    save_every: int = 1,  # Save every epoch by default 
    save_dir: str = "outputs/full_reconstruction",
    seed: int = 42
):
    """
    Train PointFlow2D on full dataset for reconstruction.
    
    Args:
        data_dir: Directory with training data
        epochs: Number of epochs
        batch_size: Batch size
        latent_dim: Latent dimension
        cnf_hidden: CNF hidden dimension
        lr: Learning rate
        beta_schedule: Beta annealing schedule
        beta_start: Starting beta value
        beta_end: Ending beta value
        save_every: Save checkpoint every N epochs
        save_dir: Directory to save outputs
        seed: Random seed for reproducibility
    """
    # Set reproducibility first
    worker_init_fn = set_seed(seed)
    
    logger = setup_logging()
    
    logger.info("🚀 POINTFLOW2D FULL RECONSTRUCTION TRAINING")
    logger.info("=" * 60)
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"🎯 Pure reconstruction test - NO validation split")
    
    # Load dataset
    logger.info("📂 Loading dataset...")
    dataset = SliceDataset(
        data_directory=data_dir,
        normalize=True,
        min_points=50  # Only filter very sparse slices
    )
    
    # Use ALL data for training (no validation split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0,  # Avoid multiprocessing issues
        worker_init_fn=worker_init_fn  # Ensure reproducible workers
    )
    
    stats = dataset.get_statistics()
    logger.info(f"✅ Dataset loaded:")
    logger.info(f"   Total slices: {stats['total_slices']:,}")
    logger.info(f"   Unique cars: {stats['unique_cars']}")
    logger.info(f"   Points per slice: {stats['min_points']} - {stats['max_points']} (avg: {stats['mean_points']:.0f})")
    logger.info(f"   Batches per epoch: {len(dataloader)}")
    
    # Create model
    logger.info("🏗️ Creating PointFlow2D model...")
    model = PointFlow2DVAE_Fixed(
        latent_dim=latent_dim,
        encoder_hidden_dim=128,
        cnf_hidden_dim=cnf_hidden,
        cnf_atol=1e-5,
        cnf_rtol=1e-5
    )
    
    # Auto-detect device (GPU first, fallback to CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"💻 Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")  # CNF needs CPU for torchdiffeq on M1
        logger.info(f"💻 MPS available but using CPU (torchdiffeq compatibility)")
    else:
        device = torch.device("cpu")
        logger.info(f"💻 Using CPU")
    
    model = model.to(device)
    
    info = model.get_model_info()
    logger.info(f"✅ Model created:")
    logger.info(f"   Total parameters: {info['total_parameters']:,}")
    logger.info(f"   Encoder: {info['encoder_parameters']:,}")
    logger.info(f"   CNF: {info['cnf_parameters']:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger.info(f"⚙️ Training configuration:")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Learning rate: {lr}")
    logger.info(f"   Beta schedule: {beta_schedule} ({beta_start} → {beta_end})")
    logger.info(f"   Save every: {save_every} epochs")
    logger.info(f"   Random seed: {seed} (REPRODUCIBLE)")
    logger.info(f"   Deterministic: {torch.backends.cudnn.deterministic}")
    
    # Create training metadata
    training_metadata = {
        'dataset_info': {
            'data_dir': data_dir,
            'total_slices': stats['total_slices'],
            'unique_cars': stats['unique_cars'],
            'min_points': stats['min_points'],
            'max_points': stats['max_points'],
            'mean_points': stats['mean_points']
        },
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'beta_schedule': beta_schedule,
            'beta_start': beta_start,
            'beta_end': beta_end,
            'optimizer': 'Adam',
            'seed': seed,
            'deterministic': torch.backends.cudnn.deterministic
        },
        'hardware_info': {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
    }
    logger.info(f"   Output directory: {save_dir}")
    
    # Training loop
    logger.info("\n🔥 Starting training...")
    model.train()
    
    losses = {
        'epoch': [],
        'recon_loss': [],
        'kl_loss': [],
        'total_loss': [],
        'beta': []
    }
    
    best_recon = float('-inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Beta schedule
        if beta_schedule == "linear":
            progress = epoch / epochs
            beta = beta_start + progress * (beta_end - beta_start)
        elif beta_schedule == "cosine":
            progress = epoch / epochs
            beta = beta_start + 0.5 * (beta_end - beta_start) * (1 + np.cos(np.pi * progress))
        else:  # constant
            beta = beta_end
        
        # Epoch training
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_total = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # Move to device
            points = batch['points'].to(device)
            mask = batch['mask'].to(device) if batch['mask'] is not None else None
            
            # Forward pass
            optimizer.zero_grad()
            loss_dict = model.compute_loss(points, mask, beta=beta)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for CNF stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            recon_loss = loss_dict['recon_loss'].item()
            kl_loss = loss_dict['kl_loss'].item()
            total_loss = loss_dict['total_loss'].item()
            
            epoch_recon += recon_loss
            epoch_kl += kl_loss
            epoch_total += total_loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'recon': f"{recon_loss:.3f}",
                'kl': f"{kl_loss:.1f}",
                'beta': f"{beta:.6f}"
            })
        
        # Epoch statistics
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        avg_total = epoch_total / num_batches
        epoch_time = time.time() - epoch_start
        
        # Track best reconstruction
        if avg_recon > best_recon:
            best_recon = avg_recon
        
        # Log epoch results
        total_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1:3d}/{epochs}: Recon={avg_recon:6.3f}, KL={avg_kl:6.1f}, Beta={beta:.6f}, Time={epoch_time:.1f}s, Total={total_time/60:.1f}min")
        
        # Store losses
        losses['epoch'].append(epoch + 1)
        losses['recon_loss'].append(avg_recon)
        losses['kl_loss'].append(avg_kl)
        losses['total_loss'].append(avg_total)
        losses['beta'].append(beta)
        
        # Save EVERY SINGLE EPOCH - complete granular control
        save_checkpoint(model, optimizer, epoch + 1, losses, save_dir, training_metadata)
        
        # Also save to latest checkpoint for easy resuming
        latest_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'model_config': {
                'latent_dim': model.latent_dim,
                'input_dim': model.input_dim,
                'encoder_hidden_dim': getattr(model, 'encoder_hidden_dim', 128),
                'cnf_hidden_dim': getattr(model, 'cnf_hidden_dim', 64),
                'total_parameters': model.get_model_info()['total_parameters'],
                'encoder_parameters': model.get_model_info()['encoder_parameters'],
                'cnf_parameters': model.get_model_info()['cnf_parameters']
            },
            'metadata': training_metadata,
            'timestamp': time.time(),
            'pytorch_version': torch.__version__
        }
        torch.save(latest_checkpoint, Path(save_dir) / "latest_checkpoint.pt")
        
        # Log saving every 5 epochs to avoid spam, but save every epoch
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(f"💾 Epoch {epoch + 1} saved (saving all epochs)")
        elif epoch == 1:
            logger.info(f"💾 Saving every epoch for granular control")
    
    # Final results
    total_time = time.time() - start_time
    logger.info("\n🎉 Training completed!")
    logger.info(f"📊 Final Results:")
    logger.info(f"   Best reconstruction loss: {best_recon:.3f}")
    logger.info(f"   Final reconstruction loss: {avg_recon:.3f}")
    logger.info(f"   Total training time: {total_time/60:.1f} minutes")
    logger.info(f"   Average time per epoch: {total_time/epochs:.1f} seconds")
    
    # Success assessment
    if best_recon > -1.0:
        logger.info("🏆 EXCELLENT: Model learned to reconstruct very well!")
        success = "excellent"
    elif best_recon > -5.0:
        logger.info("✅ GOOD: Solid reconstruction learning")
        success = "good"
    elif best_recon > -20.0:
        logger.info("🟡 PARTIAL: Some learning, could be better")
        success = "partial"
    else:
        logger.info("❌ POOR: Reconstruction learning failed")
        success = "poor"
    
    logger.info(f"\n🎯 Next Steps:")
    if success in ["excellent", "good"]:
        logger.info("   ✅ Ready for visualization and generation testing")
        logger.info("   ✅ Architecture validated for Stage 2 (Transformer)")
        logger.info(f"   ✅ Use checkpoint: {save_dir}/latest_checkpoint.pt")
    else:
        logger.info("   🔧 Need architecture or hyperparameter tuning")
        logger.info("   📊 Check loss curves for debugging")
    
    return {
        'success': success,
        'best_recon': best_recon,
        'final_recon': avg_recon,
        'total_time': total_time,
        'model_path': f"{save_dir}/latest_checkpoint.pt"
    }


def main():
    """CLI interface for full reconstruction training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PointFlow2D for full reconstruction")
    parser.add_argument("data_dir", help="Directory containing training data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--cnf-hidden", type=int, default=64, help="CNF hidden dimension")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--beta-schedule", default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--beta-start", type=float, default=0.0, help="Starting beta")
    parser.add_argument("--beta-end", type=float, default=0.01, help="Ending beta")
    parser.add_argument("--save-every", type=int, default=1, help="Save every N epochs (default: 1 = save all epochs)")
    parser.add_argument("--save-dir", default="outputs/full_reconstruction", help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Run training
    results = train_full_reconstruction(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        cnf_hidden=args.cnf_hidden,
        lr=args.lr,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        save_every=args.save_every,
        save_dir=args.save_dir,
        seed=args.seed
    )
    
    print(f"\n📋 Training Summary:")
    print(f"   Success: {results['success']}")
    print(f"   Best reconstruction: {results['best_recon']:.3f}")
    print(f"   Training time: {results['total_time']/60:.1f} minutes")
    print(f"   Model saved: {results['model_path']}")


if __name__ == "__main__":
    main()
