"""
Complete PointFlow2D VAE model combining encoder and CNF decoder.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np

# Handle both relative and absolute imports
try:
    from .encoder import PointNet2DEncoder, reparameterize, kl_divergence
    from .cnf import ContinuousNormalizingFlow
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from encoder import PointNet2DEncoder, reparameterize, kl_divergence
    from cnf import ContinuousNormalizingFlow

logger = logging.getLogger(__name__)


class PointFlow2DVAE(nn.Module):
    """
    Complete PointFlow2D Variational Autoencoder.
    
    Architecture:
    - Encoder: PointNet2D -> latent distribution (mu, logvar)
    - Decoder: CNF conditioned on latent sample -> 2D point cloud
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 latent_dim: int = 128,
                 encoder_hidden_dim: int = 256,
                 cnf_hidden_dim: int = 128,
                 use_adjoint: bool = True):
        """
        Initialize PointFlow2D VAE.
        
        Args:
            input_dim: Dimension of input points (2 for 2D)
            latent_dim: Dimension of latent space
            encoder_hidden_dim: Hidden dimension for encoder
            cnf_hidden_dim: Hidden dimension for CNF
            use_adjoint: Whether to use adjoint method in CNF
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.cnf_hidden_dim = cnf_hidden_dim
        
        # Components
        self.encoder = PointNet2DEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim
        )
        
        self.decoder = ContinuousNormalizingFlow(
            dim=input_dim,
            hidden_dim=cnf_hidden_dim,
            context_dim=latent_dim,
            use_adjoint=use_adjoint
        )
        
        logger.info(f"Initialized PointFlow2D VAE with latent_dim={latent_dim}")
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input points to latent distribution.
        
        Args:
            x: Input points, shape (batch_size, num_points, input_dim)
            mask: Optional mask for valid points
            
        Returns:
            Tuple of (mu, logvar)
        """
        return self.encoder(x, mask)
    
    def decode(self, z: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Decode latent vector to point cloud.
        
        Args:
            z: Latent vector, shape (batch_size, latent_dim)
            num_points: Number of points to generate
            
        Returns:
            Generated points, shape (batch_size, num_points, input_dim)
        """
        return self.decoder.sample(z, num_points)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                num_points_out: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input points, shape (batch_size, num_points_in, input_dim)
            mask: Optional mask for valid points
            num_points_out: Number of points to generate (default: same as input)
            
        Returns:
            Dictionary containing:
                - 'mu': Latent mean
                - 'logvar': Latent log variance
                - 'z': Sampled latent vector
                - 'x_recon': Reconstructed points
                - 'log_likelihood': Log likelihood from CNF
                - 'kl_loss': KL divergence loss
        """
        batch_size, num_points_in, _ = x.shape
        
        if num_points_out is None:
            num_points_out = num_points_in
        
        # Encode
        mu, logvar = self.encode(x, mask)
        
        # Sample latent vector
        z = reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z, num_points_out)
        
        # Compute log likelihood of reconstruction
        if num_points_out == num_points_in:
            # Can compute likelihood of original points
            log_likelihood = self.decoder.log_prob(x, z)
        else:
            # Can't directly compare, use placeholder
            log_likelihood = torch.zeros(batch_size, device=x.device)
        
        # Compute KL divergence
        kl_loss = kl_divergence(mu, logvar)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'x_recon': x_recon,
            'log_likelihood': log_likelihood,
            'kl_loss': kl_loss
        }
    
    def sample(self, batch_size: int, num_points: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample new point clouds from the prior.
        
        Args:
            batch_size: Number of samples to generate
            num_points: Number of points per sample
            device: Device to generate on
            
        Returns:
            Generated points, shape (batch_size, num_points, input_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Decode
        return self.decode(z, num_points)
    
    def reconstruct(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   num_points_out: Optional[int] = None) -> torch.Tensor:
        """
        Reconstruct input points.
        
        Args:
            x: Input points
            mask: Optional mask
            num_points_out: Number of output points
            
        Returns:
            Reconstructed points
        """
        with torch.no_grad():
            result = self.forward(x, mask, num_points_out)
            return result['x_recon']
    
    def compute_loss(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                    beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            x: Input points
            mask: Optional mask
            beta: Beta coefficient for KL loss
            
        Returns:
            Dictionary with loss components
        """
        result = self.forward(x, mask)
        
        # Reconstruction loss (negative log likelihood)
        recon_loss = -result['log_likelihood'].mean()
        
        # KL divergence loss
        kl_loss = result['kl_loss'].mean()
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': torch.tensor(beta)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        return {
            'total_parameters': total_params,
            'encoder_parameters': encoder_params,
            'decoder_parameters': decoder_params,
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'encoder_hidden_dim': self.encoder_hidden_dim,
            'cnf_hidden_dim': self.cnf_hidden_dim
        }


def main():
    """CLI interface for testing PointFlow2D VAE."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PointFlow2D VAE")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-points", type=int, default=100, help="Number of points")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--test-variable-points", action="store_true", 
                       help="Test variable point generation")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print(f"🧪 Testing PointFlow2D VAE...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Points per batch: {args.num_points}")
    print(f"   Latent dimension: {args.latent_dim}")
    
    # Create model
    model = PointFlow2DVAE(latent_dim=args.latent_dim)
    
    # Model info
    info = model.get_model_info()
    print(f"\n📊 Model Statistics:")
    for key, value in info.items():
        if isinstance(value, int):
            print(f"   {key}: {value:,}")
        else:
            print(f"   {key}: {value}")
    
    # Test data
    test_points = torch.randn(args.batch_size, args.num_points, 2)
    test_mask = torch.ones(args.batch_size, args.num_points)
    
    print(f"\n🔍 Testing forward pass...")
    try:
        # Forward pass
        result = model.forward(test_points, test_mask)
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {test_points.shape}")
        print(f"   Latent mu shape: {result['mu'].shape}")
        print(f"   Latent z shape: {result['z'].shape}")
        print(f"   Reconstruction shape: {result['x_recon'].shape}")
        print(f"   KL loss: {result['kl_loss'].mean().item():.4f}")
        
        # Test loss computation
        loss_dict = model.compute_loss(test_points, test_mask)
        print(f"\n📊 Loss computation:")
        for key, value in loss_dict.items():
            print(f"   {key}: {value.item():.4f}")
        
        # Test sampling
        print(f"\n🎲 Testing sampling...")
        samples = model.sample(args.batch_size, args.num_points)
        print(f"   Generated samples shape: {samples.shape}")
        print(f"   Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        # Test reconstruction
        print(f"\n🔄 Testing reconstruction...")
        reconstruction = model.reconstruct(test_points, test_mask)
        recon_error = (test_points - reconstruction).abs().mean()
        print(f"   Reconstruction error: {recon_error:.6f}")
        
        if args.test_variable_points:
            print(f"\n🔢 Testing variable point generation...")
            for target_points in [50, 75, 150]:
                var_recon = model.reconstruct(test_points, test_mask, target_points)
                print(f"   {args.num_points} -> {target_points} points: {var_recon.shape}")
        
        print(f"\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
