"""
PointFlow2D VAE - FINAL FIXED VERSION
All architectural issues resolved for single slice overfitting test.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import logging

def standard_normal_logprob(z):
    """
    Compute log probability of standard normal distribution.
    From original PointFlow implementation.
    """
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2

def gaussian_entropy(logvar):
    """
    Compute entropy of Gaussian distribution.
    From original PointFlow implementation.
    """
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent

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


class PointFlow2DVAE(nn.Module):
    """
    PointFlow2D VAE - FINAL FIXED VERSION.
    Complete implementation with all architectural issues resolved.
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
                 cnf_rtol: float = 1e-5,
                 force_cpu_ode: bool = False):
        """
        Initialize PointFlow2D VAE.
        
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
            force_cpu_ode: Force ODE integration on CPU (set to False for GPU!)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_latent_flow = use_latent_flow
        
        # Original PointFlow training parameters
        self.prior_weight = 1.0
        self.recon_weight = 1.0
        self.entropy_weight = 1.0
        self.use_deterministic_encoder = False  # We use variational encoder
        
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
            rtol=cnf_rtol,
            force_cpu_ode=force_cpu_ode  # Should be False for GPU!
        )
        
        # Latent CNF (transforms latent distribution)
        if use_latent_flow:
            self.latent_cnf = LatentCNF(
                latent_dim=latent_dim,
                hidden_dim=latent_cnf_hidden_dim,
                solver=cnf_solver,
                atol=cnf_atol,
                rtol=cnf_rtol,
                force_cpu_ode=force_cpu_ode  # Should be False for GPU!
            )
        else:
            self.latent_cnf = None
        
        logger.info(f"Initialized PointFlow2DVAE:")
        logger.info(f"  latent_dim={latent_dim}, use_latent_flow={use_latent_flow}")
        logger.info(f"  cnf_hidden_dim={cnf_hidden_dim}, latent_cnf_hidden_dim={latent_cnf_hidden_dim}")
        logger.info(f"  force_cpu_ode={force_cpu_ode} (should be False for GPU training!)")
    
    def forward(self, x, opt, step, writer=None):
        """
        EXACT PointFlow training forward pass.
        Handles complete training step including loss computation, backward, and optimizer step.
        
        Args:
            x: Input points, shape (batch_size, num_points, input_dim)
            opt: Optimizer
            step: Current training step
            writer: Optional tensorboard writer
            
        Returns:
            Dictionary containing training metrics
        """
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        
        # Encode to get latent distribution
        z_mu, z_sigma = self.encoder(x)
        
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = reparameterize(z_mu, z_sigma)

        # Compute H[Q(z|X)] - entropy of encoder distribution
        if self.use_deterministic_encoder:
            entropy = torch.zeros(batch_size).to(z)
        else:
            entropy = gaussian_entropy(z_sigma)

        # Compute the prior probability P(z) using Latent CNF
        if self.use_latent_flow and self.latent_cnf is not None:
            w, delta_log_pw = self.latent_cnf(
                z, None, torch.zeros(batch_size, 1).to(z)
            )
            log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_pw = delta_log_pw.view(batch_size, 1)
            log_pz = log_pw - delta_log_pw
        else:
            # FIX: When no latent flow, prior is standard normal on z
            log_pz = standard_normal_logprob(z).view(batch_size, -1).sum(1, keepdim=True)

        # Compute the reconstruction likelihood P(X|z) using Point CNF
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()  # Gradient flow trick from original
        
        # Forward pass through Point CNF: x (slice) -> y (Gaussian)
        y, delta_log_py = self.point_cnf(
            x, z_new, torch.zeros(batch_size, num_points, 1).to(x)
        )
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        # Loss computation exactly as original PointFlow
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        loss = entropy_loss + prior_loss + recon_loss
        loss.backward()
        opt.step()

        # Logging
        entropy_log = entropy.mean()
        recon = -log_px.mean()
        prior = -log_pz.mean()

        recon_nats = recon / float(x.size(1) * x.size(2))
        prior_nats = prior / float(self.latent_dim)

        if writer is not None:
            writer.add_scalar('train/entropy', entropy_log, step)
            writer.add_scalar('train/prior', prior, step)
            writer.add_scalar('train/prior(nats)', prior_nats, step)
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)

        return {
            'entropy': entropy_log.cpu().detach().item()
            if not isinstance(entropy_log, float) else entropy_log,
            'prior_nats': prior_nats.cpu().detach().item(),
            'recon_nats': recon_nats.cpu().detach().item(),
        }
    
    def encode(self, x):
        """Encode point cloud to latent representation."""
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return reparameterize(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None):
        """Decode latent code to point cloud using Point CNF."""
        # Generate points from standard normal
        y = torch.randn(z.size(0), num_points, self.input_dim).to(z)
        if truncate_std is not None:
            # Apply truncation if specified
            y = torch.clamp(y, -truncate_std, truncate_std)
        
        # Transform through Point CNF in REVERSE: Gaussian -> Slice
        x, _ = self.point_cnf(y, z, reverse=True)
        x = x.view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None):
        """Sample point clouds exactly as original PointFlow."""
        assert self.use_latent_flow, "Sampling requires `self.use_latent_flow` to be True."
        
        # Generate the shape code from the prior
        device = next(self.parameters()).device if gpu is None else f'cuda:{gpu}'
        w = torch.randn(batch_size, self.latent_dim).to(device)
        if truncate_std_latent is not None:
            w = torch.clamp(w, -truncate_std_latent, truncate_std_latent)
        
        z, _ = self.latent_cnf(w, None, reverse=True)
        z = z.view(*w.size())
        
        # Sample points conditioned on the shape code
        y = torch.randn(batch_size, num_points, self.input_dim).to(device)
        if truncate_std is not None:
            y = torch.clamp(y, -truncate_std, truncate_std)
        
        x, _ = self.point_cnf(y, z, reverse=True)
        x = x.view(*y.size())
        return z, x

    def reconstruct(self, x, num_points=None, truncate_std=None):
        """Reconstruct point clouds exactly as original PointFlow."""
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x_recon = self.decode(z, num_points, truncate_std)
        return x_recon
    
    def make_optimizer(self, args):
        """Create optimizer exactly as original PointFlow."""
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        
        # FIX: Proper parameter handling
        params = list(self.encoder.parameters()) + list(self.point_cnf.parameters())
        if self.use_latent_flow and self.latent_cnf is not None:
            params += list(self.latent_cnf.parameters())
        
        opt = _get_opt_(params)
        return opt
    
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
    """Test PointFlow2DVAE implementation."""
    print("🧪 Testing PointFlow2DVAE FINAL VERSION")
    
    # Create model with COMPLETE PointFlow architecture
    model = PointFlow2DVAE(
        latent_dim=64, 
        cnf_hidden_dim=128, 
        latent_cnf_hidden_dim=128,
        use_latent_flow=True,
        force_cpu_ode=False  # GPU MODE!
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    model = model.to(device)
    
    # Test data
    batch_size = 2
    num_points = 100
    points = torch.randn(batch_size, num_points, 2).to(device)
    
    info = model.get_model_info()
    print(f"✅ Model created: {info['total_parameters']:,} parameters")
    print(f"   Encoder: {info['encoder_parameters']:,}")
    print(f"   Point CNF: {info['point_cnf_parameters']:,}")
    print(f"   Latent CNF: {info['latent_cnf_parameters']:,}")
    print(f"   Use Latent Flow: {info['use_latent_flow']}")
    
    # FIX: Test configuration with all required attributes
    class TestArgs:
        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        weight_decay = 0.
        optimizer = 'adam'
        momentum = 0.9  # FIX: Added for SGD fallback
    
    opt_args = TestArgs()
    optimizer = model.make_optimizer(opt_args)
    print(f"✅ Optimizer created with {len(list(model.parameters()))} parameter groups")

    # Test forward pass
    result = model.forward(points, optimizer, 0)
    print(f"✅ Forward pass completed:")
    print(f"   Entropy: {result['entropy']:.3f}")
    print(f"   Prior nats: {result['prior_nats']:.3f}")
    print(f"   Recon nats: {result['recon_nats']:.3f}")
    
    # Test sampling
    if model.use_latent_flow:
        z, samples = model.sample(batch_size, num_points)
        print(f"✅ Sampling: z shape {z.shape}, samples shape {samples.shape}")
    
    # Test reconstruction
    recon_samples = model.reconstruct(points)
    print(f"✅ Reconstruction: {recon_samples.shape}")
    
    # Verify GPU usage
    if device.type == 'cuda':
        print(f"✅ Model is on GPU: {next(model.parameters()).is_cuda}")
    
    print("🎉 PointFlow2DVAE FINAL VERSION test passed!")
    print("\n📝 Ready for single slice overfitting test!")
    print("   Remember to set force_cpu_ode=False for GPU training")


if __name__ == "__main__":
    main()
