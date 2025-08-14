#!/usr/bin/env python3
"""
Complete training script for PointFlow2D VAE.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.trainer import PointFlow2DTrainer, plot_training_history
from training.dataset import SliceDataset
from models.pointflow2d import PointFlow2DVAE
import torch


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train PointFlow2D VAE")
    parser.add_argument("data_dir", help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--beta-schedule", default="linear", 
                       choices=["constant", "linear", "cosine"],
                       help="Beta annealing schedule")
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Starting beta")
    parser.add_argument("--beta-end", type=float, default=1e-2, help="Ending beta")
    parser.add_argument("--save-dir", default="outputs/training", help="Save directory")
    parser.add_argument("--min-points", type=int, default=100, help="Minimum points per slice")
    parser.add_argument("--max-points", type=int, default=2000, help="Maximum points per slice")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / 'training.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    print("🚀 PointFlow2D Training Pipeline")
    print("=" * 50)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🧠 Latent dimension: {args.latent_dim}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"🔧 Beta schedule: {args.beta_schedule} ({args.beta_start} → {args.beta_end})")
    print(f"💾 Save directory: {args.save_dir}")
    print(f"🔍 Point range: {args.min_points} - {args.max_points}")
    print("=" * 50)
    
    try:
        # Check data directory
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Create full dataset
        print(f"\n📂 Loading dataset from {args.data_dir}...")
        full_dataset = SliceDataset(
            data_directory=args.data_dir,
            normalize=True,
            min_points=args.min_points,
            max_points=args.max_points
        )
        
        stats = full_dataset.get_statistics()
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total slices: {stats['total_slices']:,}")
        print(f"   Unique cars: {stats['unique_cars']}")
        print(f"   Points per slice: {stats['min_points']} - {stats['max_points']} (avg: {stats['mean_points']:.0f})")
        
        # Split into train/val
        if args.val_split > 0:
            val_size = int(len(full_dataset) * args.val_split)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            print(f"📊 Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
        else:
            train_dataset = full_dataset
            val_dataset = None
            print(f"📊 Using full dataset for training: {len(train_dataset)} slices")
        
        # Create model
        print(f"\n🏗️ Creating PointFlow2D model...")
        model = PointFlow2DVAE(latent_dim=args.latent_dim)
        
        model_info = model.get_model_info()
        print(f"✅ Model created successfully!")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Encoder parameters: {model_info['encoder_parameters']:,}")
        print(f"   Decoder parameters: {model_info['decoder_parameters']:,}")
        
        # Create trainer
        print(f"\n🎯 Setting up trainer...")
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
        
        # Start training
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        print(f"💾 Checkpoints will be saved to: {Path(args.save_dir).absolute()}")
        
        history = trainer.train(
            num_epochs=args.epochs,
            save_dir=args.save_dir,
            save_every=10,
            validate_every=5
        )
        
        # Plot results
        print(f"\n📊 Generating training curves...")
        plot_training_history(history, f"{args.save_dir}/training_curves.png")
        
        # Final summary
        print(f"\n🎉 Training completed successfully!")
        print(f"📈 Final training loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss']:
            print(f"📉 Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"💾 Model saved to: {Path(args.save_dir).absolute()}")
        print(f"📊 Training curves saved to: {Path(args.save_dir).absolute()}/training_curves.png")
        
        print(f"\n✨ Next steps:")
        print(f"   1. Visualize results: python scripts/visualize_pointflow2d.py {args.save_dir}/final_model.pt {args.data_dir}")
        print(f"   2. Check training logs: {logs_dir}/training.log")
        print(f"   3. Analyze checkpoints in: {args.save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
