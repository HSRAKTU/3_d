#!/usr/bin/env python3
"""
Optimized overfitting test for PointFlow2D on M1 MacBook Air.
"""

import sys
import torch
import numpy as np
import logging
import time
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_fixed import PointFlow2DVAE_Fixed


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging():
    """Setup logging for the test."""
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / 'overfitting_test.log')
        ]
    )
    return logging.getLogger(__name__)


def test_overfitting(
    slice_path: str,
    epochs: int = 200,
    latent_dim: int = 32,
    cnf_hidden: int = 64,
    lr: float = 5e-5,
    beta_max: float = 0.001,
    seed: int = 42
):
    """
    Test overfitting on a single slice with optimized settings for M1.
    
    Args:
        slice_path: Path to the single slice file
        epochs: Number of training epochs
        latent_dim: Latent dimension (small for 2D)
        cnf_hidden: CNF hidden dimension (small for 2D)
        lr: Learning rate
        beta_max: Maximum beta value for KL weighting
        seed: Random seed for reproducibility
    """
    # Set reproducibility first
    set_seed(seed)
    
    logger = setup_logging()
    
    logger.info("🚀 Starting PointFlow2D Overfitting Test (M1 Optimized)")
    logger.info("=" * 60)
    
    # Load and preprocess data
    logger.info(f"📂 Loading slice from: {slice_path}")
    single_slice = np.load(slice_path)
    slice_tensor = torch.FloatTensor(single_slice).unsqueeze(0)
    
    # Normalize to [-1, 1]
    x_min, x_max = slice_tensor.min(dim=1, keepdim=True)[0], slice_tensor.max(dim=1, keepdim=True)[0]
    slice_normalized = 2 * (slice_tensor - x_min) / (x_max - x_min) - 1
    
    logger.info(f"✅ Slice loaded: {single_slice.shape[0]} points")
    logger.info(f"   Original range: X[{single_slice[:, 0].min():.3f}, {single_slice[:, 0].max():.3f}], Y[{single_slice[:, 1].min():.3f}, {single_slice[:, 1].max():.3f}]")
    logger.info(f"   Normalized range: [{slice_normalized.min():.3f}, {slice_normalized.max():.3f}]")
    
    # Create M1-optimized model
    logger.info(f"🏗️ Creating optimized model...")
    model = PointFlow2DVAE_Fixed(
        latent_dim=latent_dim,
        encoder_hidden_dim=128,  # Reasonable for 2D
        cnf_hidden_dim=cnf_hidden,
        cnf_atol=1e-5,  # Balanced tolerance
        cnf_rtol=1e-5
    )
    
    # Use CPU for compatibility with torchdiffeq float64 requirements
    device = torch.device("cpu")
    logger.info("💻 Using CPU (torchdiffeq compatibility)")
    
    model = model.to(device)
    slice_normalized = slice_normalized.to(device)
    
    info = model.get_model_info()
    logger.info(f"✅ Model created:")
    logger.info(f"   Total parameters: {info['total_parameters']:,}")
    logger.info(f"   Encoder: {info['encoder_parameters']:,}")
    logger.info(f"   CNF: {info['cnf_parameters']:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger.info(f"⚙️ Training configuration:")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Learning rate: {lr}")
    logger.info(f"   Max beta: {beta_max}")
    logger.info(f"   Device: {device}")
    
    # Training loop
    logger.info("\n🔥 Starting training...")
    model.train()
    
    best_recon = float('-inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Progressive beta schedule (focus on reconstruction first)
        beta = min(beta_max, epoch / (epochs * 0.8)) * beta_max
        
        # Forward pass
        loss_dict = model.compute_loss(slice_normalized, beta=beta)
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Logging
        recon_loss = loss_dict['recon_loss'].item()
        kl_loss = loss_dict['kl_loss'].item()
        total_loss = loss_dict['total_loss'].item()
        
        if recon_loss > best_recon:
            best_recon = recon_loss
        
        # Log progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch:3d}/{epochs}: Recon={recon_loss:6.3f}, KL={kl_loss:6.1f}, Beta={beta:.6f}, Time={elapsed:.1f}s")
    
    # Final evaluation
    logger.info("\n📊 Final Results:")
    logger.info(f"   Best reconstruction loss: {best_recon:.3f}")
    logger.info(f"   Final reconstruction loss: {recon_loss:.3f}")
    logger.info(f"   Total training time: {time.time() - start_time:.1f}s")
    
    # Test generation quality
    logger.info("\n🎲 Testing generation quality...")
    model.eval()
    
    try:
        with torch.no_grad():
            mu, logvar = model.encode(slice_normalized)
            
            # Conservative generation
            generated = model.decode(mu * 0.5, single_slice.shape[0], temperature=0.1)
            
            # Denormalize for comparison
            generated_denorm = (generated + 1) * (x_max - x_min) / 2 + x_min
            original_denorm = (slice_normalized + 1) * (x_max - x_min) / 2 + x_min
            
            # Quality metrics
            mse = torch.mean((generated_denorm[0] - original_denorm[0])**2).item()
            
            logger.info(f"   Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
            logger.info(f"   MSE (denormalized): {mse:.6f}")
            
            # Success assessment
            if best_recon > -1.0:
                logger.info("🏆 EXCELLENT: Ready for full dataset training!")
                success = "excellent"
            elif best_recon > -10.0:
                logger.info("✅ GOOD: Architecture validated, needs minor tuning")
                success = "good"
            elif best_recon > -100.0:
                logger.info("🟡 PARTIAL: Some learning, needs more work")
                success = "partial"
            else:
                logger.info("❌ POOR: Significant issues remain")
                success = "poor"
                
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        success = "failed"
    
    logger.info("\n🎯 Conclusion:")
    logger.info(f"   Overfitting test: {success}")
    logger.info(f"   Model size: Optimized for M1 ({info['total_parameters']:,} params)")
    logger.info(f"   Ready for full training: {'Yes' if success in ['excellent', 'good'] else 'Needs work'}")
    
    return {
        'success': success,
        'best_recon': best_recon,
        'final_recon': recon_loss,
        'total_params': info['total_parameters'],
        'training_time': time.time() - start_time
    }


def main():
    """CLI interface for overfitting test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PointFlow2D overfitting")
    parser.add_argument("slice_path", help="Path to single slice .npy file")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--cnf-hidden", type=int, default=64, help="CNF hidden dimension")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--beta-max", type=float, default=0.001, help="Maximum beta")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    results = test_overfitting(
        slice_path=args.slice_path,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        cnf_hidden=args.cnf_hidden,
        lr=args.lr,
        beta_max=args.beta_max,
        seed=args.seed
    )
    
    print(f"\n📋 Summary:")
    print(f"   Success: {results['success']}")
    print(f"   Best reconstruction: {results['best_recon']:.3f}")
    print(f"   Parameters: {results['total_params']:,}")
    print(f"   Training time: {results['training_time']:.1f}s")


if __name__ == "__main__":
    main()
