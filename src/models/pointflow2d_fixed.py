"""
PointFlow2D VAE with corrected CNF implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import logging

try:
    from .encoder import PointNet2DEncoder, reparameterize, kl_divergence
    from .pointflow_cnf import PointFlowCNF
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from encoder import PointNet2DEncoder, reparameterize, kl_divergence
    from pointflow_cnf import PointFlowCNF

logger = logging.getLogger(__name__)


class PointFlow2DVAE_Fixed(nn.Module):
    """
    PointFlow2D VAE with exact PointFlow CNF implementation.
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 latent_dim: int = 128,
                 encoder_hidden_dim: int = 256,
                 cnf_hidden_dim: int = 128,
                 cnf_solver: str = 'dopri5',
                 cnf_atol: float = 1e-5,
                 cnf_rtol: float = 1e-5):
        """
        Initialize PointFlow2D VAE.
        
        Args:
            input_dim: Dimension of input points (2 for 2D)
            latent_dim: Dimension of latent space
            encoder_hidden_dim: Hidden dimension for encoder
            cnf_hidden_dim: Hidden dimension for CNF
            cnf_solver: ODE solver for CNF
            cnf_atol: Absolute tolerance for ODE solver
            cnf_rtol: Relative tolerance for ODE solver
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder (PointNet2D)
        self.encoder = PointNet2DEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim
        )
        
        # Decoder (PointFlow CNF)
        self.cnf = PointFlowCNF(
            point_dim=input_dim,
            context_dim=latent_dim,
            hidden_dim=cnf_hidden_dim,
            solver=cnf_solver,
            atol=cnf_atol,
            rtol=cnf_rtol
        )
        
        logger.info(f"Initialized PointFlow2DVAE_Fixed with latent_dim={latent_dim}")
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input points to latent distribution.
        
        Args:
            x: Input points, shape (batch_size, num_points, input_dim)
            mask: Optional mask for valid points
            
        Returns:
            (mu, logvar) - latent distribution parameters
        """
        return self.encoder(x, mask)
    
    def decode(self, z: torch.Tensor, num_points: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Decode latent code to points.
        
        Args:
            z: Latent codes, shape (batch_size, latent_dim)
            num_points: Number of points to generate
            temperature: Sampling temperature
            
        Returns:
            Generated points, shape (batch_size, num_points, input_dim)
        """
        return self.cnf.sample(z, num_points, temperature)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input points, shape (batch_size, num_points, input_dim)
            mask: Optional mask for valid points
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode
        mu, logvar = self.encode(x, mask)
        
        # Reparameterize
        z = reparameterize(mu, logvar)
        
        # Compute log probability using CNF
        log_likelihood = self.cnf.log_prob(x, z)
        
        # Handle masking
        if mask is not None:
            log_likelihood = log_likelihood * mask
            # Sum over valid points only
            log_likelihood = log_likelihood.sum(dim=1) / mask.sum(dim=1)
        else:
            # Average over all points
            log_likelihood = log_likelihood.mean(dim=1)
        
        # KL divergence
        kl_loss = kl_divergence(mu, logvar)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'log_likelihood': log_likelihood,
            'kl_loss': kl_loss
        }
    
    def sample(self, 
               batch_size: int, 
               num_points: int, 
               device: Optional[torch.device] = None,
               temperature: float = 1.0) -> torch.Tensor:
        """
        Sample new point clouds.
        
        Args:
            batch_size: Number of samples
            num_points: Number of points per sample
            device: Device to generate on
            temperature: Sampling temperature
            
        Returns:
            Generated points, shape (batch_size, num_points, input_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Decode
        return self.decode(z, num_points, temperature)
    
    def compute_loss(self, 
                     x: torch.Tensor, 
                     mask: Optional[torch.Tensor] = None,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            x: Input points
            mask: Optional mask
            beta: Beta parameter for KL weighting
            
        Returns:
            Dictionary containing losses
        """
        result = self.forward(x, mask)
        
        # Reconstruction loss (negative log likelihood)
        recon_loss = -result['log_likelihood'].mean()
        
        # KL loss
        kl_loss = result['kl_loss'].mean()
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def get_model_info(self) -> Dict[str, int]:
        """Get model parameter information."""
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        cnf_params = sum(p.numel() for p in self.cnf.parameters())
        
        return {
            'total_parameters': total_params,
            'encoder_parameters': encoder_params,
            'cnf_parameters': cnf_params
        }


def main():
    """Test PointFlow2DVAE_Fixed implementation."""
    print("🧪 Testing PointFlow2DVAE_Fixed")
    
    # Create model
    model = PointFlow2DVAE_Fixed(latent_dim=64, cnf_hidden_dim=128)
    
    # Test data
    batch_size = 2
    num_points = 100
    points = torch.randn(batch_size, num_points, 2)
    
    info = model.get_model_info()
    print(f"✅ Model created: {info['total_parameters']:,} parameters")
    print(f"   Encoder: {info['encoder_parameters']:,}")
    print(f"   CNF: {info['cnf_parameters']:,}")
    
    # Test forward pass
    result = model.forward(points)
    print(f"✅ Forward pass: log_likelihood shape {result['log_likelihood'].shape}")
    print(f"   Log likelihood range: [{result['log_likelihood'].min():.3f}, {result['log_likelihood'].max():.3f}]")
    
    # Test loss computation
    loss_dict = model.compute_loss(points, beta=0.01)
    print(f"✅ Loss computation:")
    print(f"   Total: {loss_dict['total_loss'].item():.3f}")
    print(f"   Recon: {loss_dict['recon_loss'].item():.3f}")
    print(f"   KL: {loss_dict['kl_loss'].item():.3f}")
    
    # Test sampling
    samples = model.sample(batch_size, num_points)
    print(f"✅ Sampling: {samples.shape}")
    
    print("🎉 PointFlow2DVAE_Fixed test passed!")


if __name__ == "__main__":
    main()
