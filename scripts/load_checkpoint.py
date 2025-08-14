#!/usr/bin/env python3
"""
Utility to load and inspect model checkpoints for inference and visualization.
"""

import sys
import torch
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_fixed import PointFlow2DVAE_Fixed


def load_checkpoint(checkpoint_path: str, device: str = "auto"):
    """
    Load a model checkpoint for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ("auto", "cpu", "cuda")
    
    Returns:
        tuple: (model, checkpoint_info)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    config = checkpoint['model_config']
    
    # Create model
    model = PointFlow2DVAE_Fixed(
        latent_dim=config['latent_dim'],
        input_dim=config['input_dim'],
        encoder_hidden_dim=config.get('encoder_hidden_dim', 128),
        cnf_hidden_dim=config.get('cnf_hidden_dim', 64)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare checkpoint info
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'model_config': config,
        'training_losses': checkpoint['losses'],
        'metadata': checkpoint.get('metadata', {}),
        'timestamp': checkpoint.get('timestamp', 0),
        'device': device
    }
    
    return model, checkpoint_info


def list_available_checkpoints(save_dir: str, show_details: bool = True):
    """List all available checkpoints in a directory with performance info."""
    save_path = Path(save_dir)
    if not save_path.exists():
        print(f"❌ Directory not found: {save_dir}")
        return []
    
    # Find all checkpoint files
    checkpoints = []
    for pattern in ["checkpoint_epoch_*.pt", "latest_checkpoint.pt"]:
        checkpoints.extend(save_path.glob(pattern))
    
    # Also check subdirectories
    for subdir in ["emergency_checkpoints", "autosave"]:
        subpath = save_path / subdir
        if subpath.exists():
            for pattern in ["checkpoint_epoch_*.pt", "latest_checkpoint.pt"]:
                checkpoints.extend(subpath.glob(pattern))
    
    checkpoints = sorted(list(set(checkpoints)))
    
    if show_details and checkpoints:
        print(f"\n📊 Checkpoint Performance Summary:")
        print("─" * 80)
        print(f"{'Epoch':>5} | {'Recon Loss':>10} | {'KL Loss':>8} | {'Size':>6} | {'Path'}")
        print("─" * 80)
        
        best_recon = float('-inf')
        best_epoch = None
        
        for cp in checkpoints:
            if "latest_checkpoint" in cp.name:
                continue
                
            try:
                checkpoint = torch.load(cp, map_location="cpu")
                epoch = checkpoint['epoch']
                losses = checkpoint['losses']
                
                if losses['recon_loss'] and epoch <= len(losses['recon_loss']):
                    recon = losses['recon_loss'][epoch-1]
                    kl = losses['kl_loss'][epoch-1] if losses['kl_loss'] and epoch <= len(losses['kl_loss']) else 0
                    
                    if recon > best_recon:
                        best_recon = recon
                        best_epoch = epoch
                    
                    size_mb = cp.stat().st_size / (1024 * 1024)
                    marker = " 🏆" if epoch == best_epoch else ""
                    
                    print(f"{epoch:>5} | {recon:>10.3f} | {kl:>8.1f} | {size_mb:>5.1f}M | {cp.name}{marker}")
            except:
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(f"{'?':>5} | {'ERROR':>10} | {'ERROR':>8} | {size_mb:>5.1f}M | {cp.name}")
        
        print("─" * 80)
        if best_epoch:
            print(f"🏆 Best reconstruction: Epoch {best_epoch} (loss: {best_recon:.3f})")
    
    return checkpoints


def inspect_checkpoint(checkpoint_path: str):
    """Print detailed information about a checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"🕐 Epoch: {checkpoint['epoch']}")
        
        if 'timestamp' in checkpoint:
            timestamp = datetime.fromtimestamp(checkpoint['timestamp'])
            print(f"📅 Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model info
        config = checkpoint['model_config']
        print(f"\n🏗️ Model Architecture:")
        print(f"   Latent dim: {config['latent_dim']}")
        print(f"   Input dim: {config['input_dim']}")
        print(f"   Total parameters: {config.get('total_parameters', 'unknown'):,}")
        
        # Training progress
        losses = checkpoint['losses']
        if losses['recon_loss']:
            final_recon = losses['recon_loss'][-1]
            best_recon = max(losses['recon_loss'])
            final_kl = losses['kl_loss'][-1] if losses['kl_loss'] else 0
            
            print(f"\n📊 Training Progress:")
            print(f"   Final reconstruction: {final_recon:.3f}")
            print(f"   Best reconstruction: {best_recon:.3f}")
            print(f"   Final KL loss: {final_kl:.1f}")
            print(f"   Total epochs trained: {len(losses['recon_loss'])}")
        
        # Metadata
        if 'metadata' in checkpoint and checkpoint['metadata']:
            metadata = checkpoint['metadata']
            if 'dataset_info' in metadata:
                dataset = metadata['dataset_info']
                print(f"\n📂 Dataset Info:")
                print(f"   Total slices: {dataset.get('total_slices', 'unknown')}")
                print(f"   Unique cars: {dataset.get('unique_cars', 'unknown')}")
            
            if 'training_config' in metadata:
                config = metadata['training_config']
                print(f"\n⚙️ Training Config:")
                print(f"   Learning rate: {config.get('learning_rate', 'unknown')}")
                print(f"   Batch size: {config.get('batch_size', 'unknown')}")
                print(f"   Beta schedule: {config.get('beta_schedule', 'unknown')}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None


def main():
    """CLI interface for checkpoint utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and inspect model checkpoints")
    parser.add_argument("command", choices=["list", "inspect", "load"], 
                       help="Command to execute")
    parser.add_argument("--save-dir", default="outputs/full_reconstruction",
                       help="Directory containing checkpoints")
    parser.add_argument("--checkpoint", help="Specific checkpoint file to inspect/load")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device for loading model")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print(f"🔍 Searching for checkpoints in: {args.save_dir}")
        checkpoints = list_available_checkpoints(args.save_dir)
        
        if not checkpoints:
            print("❌ No checkpoints found")
            return
        
        print(f"\n📚 Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            size_mb = cp.stat().st_size / (1024 * 1024)
            print(f"   📁 {cp.relative_to(Path(args.save_dir).parent)} ({size_mb:.1f} MB)")
    
    elif args.command == "inspect":
        if not args.checkpoint:
            print("❌ Please specify --checkpoint for inspect command")
            return
        
        inspect_checkpoint(args.checkpoint)
    
    elif args.command == "load":
        if not args.checkpoint:
            print("❌ Please specify --checkpoint for load command")
            return
        
        print(f"🚀 Loading checkpoint: {args.checkpoint}")
        try:
            model, info = load_checkpoint(args.checkpoint, args.device)
            print(f"✅ Model loaded successfully on {info['device']}")
            print(f"📊 Epoch {info['epoch']}, Parameters: {info['model_config'].get('total_parameters', 'unknown'):,}")
            print(f"🎯 Ready for inference!")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")


if __name__ == "__main__":
    main()
