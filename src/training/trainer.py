"""
PointFlow2D training pipeline with visualization and analysis tools.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import time
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.pointflow2d import PointFlow2DVAE
from training.dataset import SliceDataset, collate_variable_length

logger = logging.getLogger(__name__)


class PointFlow2DTrainer:
    """
    Complete training pipeline for PointFlow2D VAE.
    """
    
    def __init__(self, 
                 model: PointFlow2DVAE,
                 train_dataset: SliceDataset,
                 val_dataset: Optional[SliceDataset] = None,
                 batch_size: int = 4,
                 learning_rate: float = 1e-3,
                 beta_schedule: str = "constant",
                 beta_start: float = 1e-4,
                 beta_end: float = 1e-2,
                 device: str = "auto"):
        """
        Initialize trainer.
        
        Args:
            model: PointFlow2D VAE model
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate
            beta_schedule: Beta annealing schedule ("constant", "linear", "cosine")
            beta_start: Starting beta value
            beta_end: Ending beta value
            device: Device to train on
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_variable_length,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_variable_length,
                num_workers=0
            )
        else:
            self.val_loader = None
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.train_losses = []
        self.val_losses = []
        self.beta_values = []
        
        logger.info(f"Initialized trainer on {self.device}")
        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
    
    def get_beta(self, epoch: int, total_epochs: int) -> float:
        """Get beta value for current epoch."""
        if self.beta_schedule == "constant":
            return self.beta_end
        elif self.beta_schedule == "linear":
            alpha = min(epoch / total_epochs, 1.0)
            return self.beta_start + alpha * (self.beta_end - self.beta_start)
        elif self.beta_schedule == "cosine":
            alpha = min(epoch / total_epochs, 1.0)
            return self.beta_start + 0.5 * (self.beta_end - self.beta_start) * (1 + np.cos(np.pi * alpha))
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        beta = self.get_beta(epoch, total_epochs)
        self.beta_values.append(beta)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch in pbar:
            # Move to device
            points = batch['points'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(points, mask, beta=beta)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1
            self.step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'recon': f"{loss_dict['recon_loss'].item():.4f}",
                'kl': f"{loss_dict['kl_loss'].item():.4f}",
                'beta': f"{beta:.6f}"
            })
        
        # Return average losses
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'beta': beta
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                points = batch['points'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                loss_dict = self.model.compute_loss(points, mask, beta=1.0)
                
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_kl_loss += loss_dict['kl_loss'].item()
                num_batches += 1
        
        return {
            'val_total_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_kl_loss': total_kl_loss / num_batches
        }
    
    def train(self, num_epochs: int, 
              save_dir: str = "outputs/training",
              save_every: int = 10,
              validate_every: int = 5) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints and logs
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
            
        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting training for {num_epochs} epochs")
        print(f"💾 Checkpoints will be saved to: {save_path}")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': [],
            'beta': []
        }
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch, num_epochs)
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_recon_loss'].append(train_metrics['recon_loss'])
            history['train_kl_loss'].append(train_metrics['kl_loss'])
            history['beta'].append(train_metrics['beta'])
            
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Recon Loss: {train_metrics['recon_loss']:.4f}")
            print(f"  KL Loss: {train_metrics['kl_loss']:.4f}")
            print(f"  Beta: {train_metrics['beta']:.6f}")
            
            # Validate
            if self.val_loader and (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                history['val_loss'].append(val_metrics['val_total_loss'])
                history['val_recon_loss'].append(val_metrics['val_recon_loss'])
                history['val_kl_loss'].append(val_metrics['val_kl_loss'])
                
                print(f"  Val Loss: {val_metrics['val_total_loss']:.4f}")
                print(f"  Val Recon: {val_metrics['val_recon_loss']:.4f}")
                print(f"  Val KL: {val_metrics['val_kl_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch+1:03d}.pt")
        
        # Final save
        self.save_checkpoint(save_path / "final_model.pt")
        
        # Save training history
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
        
        return history
    
    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'beta_values': self.beta_values
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.beta_values = checkpoint.get('beta_values', [])
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")


def plot_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    if 'val_loss' in history and history['val_loss']:
        val_epochs = np.arange(4, len(history['train_loss']), 5)  # Assuming validate_every=5
        axes[0, 0].plot(val_epochs, history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(history['train_recon_loss'], label='Train Recon', alpha=0.8)
    if 'val_recon_loss' in history and history['val_recon_loss']:
        val_epochs = np.arange(4, len(history['train_recon_loss']), 5)
        axes[0, 1].plot(val_epochs, history['val_recon_loss'], label='Val Recon', alpha=0.8)
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL loss
    axes[1, 0].plot(history['train_kl_loss'], label='Train KL', alpha=0.8)
    if 'val_kl_loss' in history and history['val_kl_loss']:
        val_epochs = np.arange(4, len(history['train_kl_loss']), 5)
        axes[1, 0].plot(val_epochs, history['val_kl_loss'], label='Val KL', alpha=0.8)
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Beta schedule
    axes[1, 1].plot(history['beta'], label='Beta', alpha=0.8, color='orange')
    axes[1, 1].set_title('Beta Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Beta Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to: {save_path}")
    
    plt.show()


def main():
    """CLI interface for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PointFlow2D VAE")
    parser.add_argument("data_dir", help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--beta-schedule", default="linear", choices=["constant", "linear", "cosine"])
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Starting beta")
    parser.add_argument("--beta-end", type=float, default=1e-2, help="Ending beta")
    parser.add_argument("--save-dir", default="outputs/training", help="Save directory")
    parser.add_argument("--min-points", type=int, default=50, help="Minimum points per slice")
    parser.add_argument("--max-points", type=int, help="Maximum points per slice")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print(f"🚀 Starting PointFlow2D training...")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🧠 Latent dimension: {args.latent_dim}")
    
    try:
        # Create full dataset
        full_dataset = SliceDataset(
            data_directory=args.data_dir,
            normalize=True,
            min_points=args.min_points,
            max_points=args.max_points
        )
        
        print(f"📊 Dataset loaded: {len(full_dataset)} slices")
        
        # Split into train/val
        if args.val_split > 0:
            val_size = int(len(full_dataset) * args.val_split)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        else:
            train_dataset = full_dataset
            val_dataset = None
            print(f"📊 Training on full dataset: {len(train_dataset)}")
        
        # Create model
        model = PointFlow2DVAE(latent_dim=args.latent_dim)
        
        # Create trainer
        trainer = PointFlow2DTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            beta_schedule=args.beta_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end
        )
        
        # Train
        history = trainer.train(
            num_epochs=args.epochs,
            save_dir=args.save_dir,
            save_every=10,
            validate_every=5
        )
        
        # Plot results
        plot_training_history(history, f"{args.save_dir}/training_curves.png")
        
        print(f"✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
