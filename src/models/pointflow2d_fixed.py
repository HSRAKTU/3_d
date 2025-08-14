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
    from .latent_cnf import LatentCNF
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from encoder import PointNet2DEncoder, reparameterize, kl_divergence
    from pointflow_cnf import PointFlowCNF
    from latent_cnf import LatentCNF

logger = logging.getLogger(__name__)


class PointFlow2DVAE_Fixed(nn.Module):
    """
    PointFlow2D VAE with COMPLETE PointFlow implementation.
    Now includes both Point CNF and Latent CNF exactly as PointFlow.
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 latent_dim: int = 128,
                 encoder_hidden_dim: int = 256,
                 cnf_hidden_dim: int = 128,
                 latent_cnf_hidden_dim: int = 128,
                 use_latent_flow: bool = True,
                 cnf_solver: str = 'dopri5',
                 cnf_atol: float = 1e-5,
                 cnf_rtol: float = 1e-5):
        """
        Initialize PointFlow2D VAE with COMPLETE architecture.
        
        Args:
            input_dim: Dimension of input points (2 for 2D)
            latent_dim: Dimension of latent space
            encoder_hidden_dim: Hidden dimension for encoder
            cnf_hidden_dim: Hidden dimension for Point CNF
            latent_cnf_hidden_dim: Hidden dimension for Latent CNF
            use_latent_flow: Whether to use Latent CNF (True for complete PointFlow)
            cnf_solver: ODE solver for CNF
            cnf_atol: Absolute tolerance for ODE solver
            cnf_rtol: Relative tolerance for ODE solver
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_latent_flow = use_latent_flow
        
        # Encoder (PointNet2D)
        self.encoder = PointNet2DEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim
        )
        
        # Point CNF (transforms points given latent)
        self.point_cnf = PointFlowCNF(
            point_dim=input_dim,
            context_dim=latent_dim,
            hidden_dim=cnf_hidden_dim,
            solver=cnf_solver,
            atol=cnf_atol,
            rtol=cnf_rtol
        )
        
        # Latent CNF (transforms latent distribution) - MISSING PIECE!
        if use_latent_flow:
            self.latent_cnf = LatentCNF(
                latent_dim=latent_dim,
                hidden_dim=latent_cnf_hidden_dim,
                solver=cnf_solver,
                atol=cnf_atol,
                rtol=cnf_rtol
            )
        else:
            self.latent_cnf = None
        
        logger.info(f"Initialized COMPLETE PointFlow2DVAE_Fixed:")
        logger.info(f"  latent_dim={latent_dim}, use_latent_flow={use_latent_flow}")
        logger.info(f"  cnf_hidden_dim={cnf_hidden_dim}, latent_cnf_hidden_dim={latent_cnf_hidden_dim}")
    
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
        return self.point_cnf.sample(z, num_points, temperature)
    
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
        
        # Compute prior probability P(z) using Latent CNF (EXACTLY AS POINTFLOW)
        if self.use_latent_flow and self.latent_cnf is not None:
            # Transform z through Latent CNF to get log P(z)
            w, log_prob_cnf = self.latent_cnf.forward(z, reverse=False)
            
            # Standard Gaussian log probability of w
            log_prob_w = -0.5 * (w ** 2).sum(dim=1) - 0.5 * self.latent_dim * np.log(2 * np.pi)
            
            # Total prior log probability
            log_prior = log_prob_w + log_prob_cnf
        else:
            # Standard VAE: assume z ~ N(0,1)
            w = None
            log_prior = torch.zeros(z.shape[0], device=z.device)
        
        # Compute reconstruction likelihood P(X|z) using Point CNF
        log_likelihood = self.point_cnf.log_prob(x, z)
        
        # Handle masking
        if mask is not None:
            log_likelihood = log_likelihood * mask
            # Sum over valid points only
            log_likelihood = log_likelihood.sum(dim=1) / mask.sum(dim=1)
        else:
            # Average over all points
            log_likelihood = log_likelihood.mean(dim=1)
        
        # KL divergence (for encoder regularization)
        kl_loss = kl_divergence(mu, logvar)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'w': w if self.use_latent_flow else None,
            'log_likelihood': log_likelihood,
            'log_prior': log_prior,
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
        
        # Sample latent codes exactly as PointFlow does
        if self.use_latent_flow and self.latent_cnf is not None:
            # Sample w ~ N(0,1) and transform through Latent CNF to get z
            w = torch.randn(batch_size, self.latent_dim, device=device)
            z, _ = self.latent_cnf.forward(w, reverse=True)
        else:
            # Standard VAE sampling
            z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Decode using Point CNF
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
        
        # Reconstruction loss (negative log likelihood from Point CNF)
        recon_loss = -result['log_likelihood'].mean()
        
        # Prior loss (negative log probability from Latent CNF)
        prior_loss = -result['log_prior'].mean()
        
        # KL loss (encoder regularization)
        kl_loss = result['kl_loss'].mean()
        
        # Total loss (EXACTLY as PointFlow)
        if self.use_latent_flow:
            # Complete PointFlow loss: reconstruction + prior + KL regularization
            total_loss = recon_loss + prior_loss + beta * kl_loss
        else:
            # Standard VAE loss
            total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'prior_loss': prior_loss,
            'kl_loss': kl_loss
        }
    
    def get_model_info(self) -> Dict[str, int]:
        """Get model parameter information."""
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        point_cnf_params = sum(p.numel() for p in self.point_cnf.parameters())
        
        if self.use_latent_flow and self.latent_cnf is not None:
            latent_cnf_params = sum(p.numel() for p in self.latent_cnf.parameters())
        else:
            latent_cnf_params = 0
        
        return {
            'total_parameters': total_params,
            'encoder_parameters': encoder_params,
            'point_cnf_parameters': point_cnf_params,
            'latent_cnf_parameters': latent_cnf_params,
            'use_latent_flow': self.use_latent_flow
        }


def main():
    """Test PointFlow2DVAE_Fixed implementation."""
    print("🧪 Testing PointFlow2DVAE_Fixed")
    
    # Create model with COMPLETE PointFlow architecture
    model = PointFlow2DVAE_Fixed(
        latent_dim=64, 
        cnf_hidden_dim=128, 
        latent_cnf_hidden_dim=128,
        use_latent_flow=True
    )
    
    # Test data
    batch_size = 2
    num_points = 100
    points = torch.randn(batch_size, num_points, 2)
    
    info = model.get_model_info()
    print(f"✅ COMPLETE Model created: {info['total_parameters']:,} parameters")
    print(f"   Encoder: {info['encoder_parameters']:,}")
    print(f"   Point CNF: {info['point_cnf_parameters']:,}")
    print(f"   Latent CNF: {info['latent_cnf_parameters']:,}")
    print(f"   Use Latent Flow: {info['use_latent_flow']}")
    
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
