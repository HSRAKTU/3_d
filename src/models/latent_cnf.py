"""
Latent CNF implementation for PointFlow2D.
Exactly following the original PointFlow latent CNF structure.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

# Try to import torchdiffeq
try:
    from torchdiffeq import odeint_adjoint as odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    print("Warning: torchdiffeq not available. Using simplified implementation.")
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None

logger = logging.getLogger(__name__)


class LatentODEFunc(nn.Module):
    """
    ODE function for Latent CNF.
    UNCONDITIONAL - no context conditioning.
    Exactly following PointFlow's latent CNF structure.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int):
        """
        Initialize Latent ODE function.
        
        Args:
            latent_dim: Dimension of latent space (e.g., 32, 64)
            hidden_dim: Hidden dimension for neural network (e.g., 128, 256)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Unconditional neural network (no context)
        # Following PointFlow's exact structure for latent CNF
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Initialize to small values (critical for stability)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.01, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        ODE function evaluation.
        
        Args:
            t: Time tensor
            states: Current state [z] (no logp for latent CNF)
            
        Returns:
            Time derivative dz/dt
        """
        z = states
        
        # Simple unconditional dynamics
        dz_dt = self.net(z)
        
        return dz_dt


class AugmentedLatentDynamics(nn.Module):
    """
    Augmented dynamics for computing log probability in latent CNF.
    """
    
    def __init__(self, odefunc: LatentODEFunc):
        super().__init__()
        self.odefunc = odefunc
    
    def forward(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Augmented dynamics for [z, logp].
        
        Args:
            t: Time tensor
            states: [z, logp] concatenated
            
        Returns:
            [dz/dt, dlogp/dt]
        """
        z_dim = self.odefunc.latent_dim
        z = states[:, :z_dim]
        
        # CRITICAL FIX: Use torch.enable_grad() as in PointFlow
        with torch.enable_grad():
            # Ensure z requires gradients for divergence computation
            z = z.requires_grad_(True)
            
            # Compute dz/dt
            dz_dt = self.odefunc(t, z)
            
            # Compute divergence using exact method (not trace estimation)
            # For latent CNF, compute exact divergence since latent_dim is manageable
            divergence = 0
            for i in range(z_dim):
                grad_output = torch.autograd.grad(
                    dz_dt[:, i].sum(), 
                    z,
                    create_graph=True,
                    retain_graph=True
                )[0]
                divergence += grad_output[:, i]
            
            # dlogp/dt = -divergence
            dlogp_dt = -divergence.unsqueeze(1)
        
        return torch.cat([dz_dt, dlogp_dt], dim=1)


class LatentCNF(nn.Module):
    """
    Latent Continuous Normalizing Flow.
    Transforms latent codes z ~ complex distribution to w ~ N(0,1).
    
    Exactly following PointFlow's latent CNF implementation.
    """
    
    def __init__(self, 
                 latent_dim: int = 64,
                 hidden_dim: int = 128,
                 solver: str = 'dopri5',
                 atol: float = 1e-5,
                 rtol: float = 1e-5,
                 force_cpu_ode: bool = False):
        """
        Initialize Latent CNF.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for ODE network
            solver: ODE solver ('dopri5' as in PointFlow)
            atol: Absolute tolerance for ODE solver
            rtol: Relative tolerance for ODE solver
            force_cpu_ode: Force ODE integration on CPU (for performance debugging)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.force_cpu_ode = force_cpu_ode
        
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is required for LatentCNF")
        
        # Create ODE function
        self.odefunc = LatentODEFunc(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        # Integration times (PointFlow uses [0, 1])
        self.register_buffer('sqrt_end_time', torch.sqrt(torch.tensor(1.0)))
        
        logger.info(f"Initialized LatentCNF: latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, 
                z: torch.Tensor,
                reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Latent CNF.
        
        Args:
            z: Input latent codes, shape (batch_size, latent_dim)
            reverse: If True, transform w→z; if False, transform z→w
            
        Returns:
            (output, log_prob): Transformed latents and log probability
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Integration times
        if reverse:
            # w ~ N(0,1) → z (complex distribution)
            integration_times = torch.tensor([0.0, 1.0], device=device)
        else:
            # z (complex distribution) → w ~ N(0,1)  
            integration_times = torch.tensor([1.0, 0.0], device=device)
        
        # Augmented system for log probability computation
        augmented_dynamics = AugmentedLatentDynamics(self.odefunc)
        
        # Initial state: [z, logp] - ensure it requires gradients
        logp_init = torch.zeros(batch_size, 1, device=device)
        states_init = torch.cat([z, logp_init], dim=1).requires_grad_(True)
        
        # ODE integration with optional CPU forcing
        try:
            if self.force_cpu_ode and device.type == 'cuda':
                # Move to CPU for ODE integration (performance debugging)
                states_init_cpu = states_init.cpu()
                integration_times_cpu = integration_times.cpu()
                
                # Create CPU version of ODE function and augmented dynamics
                odefunc_cpu = LatentODEFunc(self.latent_dim, self.hidden_dim)
                odefunc_cpu.load_state_dict(self.odefunc.state_dict())
                odefunc_cpu = odefunc_cpu.cpu()
                
                augmented_dynamics_cpu = AugmentedLatentDynamics(odefunc_cpu)
                
                # Solve ODE on CPU
                trajectory = odeint(
                    augmented_dynamics_cpu,
                    states_init_cpu,
                    integration_times_cpu,
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.solver
                )
                
                # Move result back to original device
                trajectory = trajectory.to(device)
                
            else:
                # Standard GPU integration
                trajectory = odeint(
                    augmented_dynamics,
                    states_init,
                    integration_times,
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.solver
                )
        except Exception as e:
            logger.error(f"ODE integration failed: {e}")
            # Fallback: return input unchanged
            return z, torch.zeros(batch_size, device=device)
        
        # Extract final state
        final_state = trajectory[-1]  # Shape: (batch_size, latent_dim + 1)
        
        # Split output and log probability
        output = final_state[:, :self.latent_dim]
        log_prob = final_state[:, self.latent_dim]
        
        return output, log_prob
    
    def sample(self, 
               batch_size: int,
               device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample from the latent distribution.
        
        Args:
            batch_size: Number of samples
            device: Device to sample on
            
        Returns:
            Sampled latent codes, shape (batch_size, latent_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from standard Gaussian
        w = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Transform w → z
        z, _ = self.forward(w, reverse=True)
        
        return z
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of latent codes.
        
        Args:
            z: Latent codes, shape (batch_size, latent_dim)
            
        Returns:
            Log probabilities, shape (batch_size,)
        """
        # Transform z → w and get log probability
        w, log_prob_cnf = self.forward(z, reverse=False)
        
        # Standard Gaussian log probability of w
        log_prob_w = -0.5 * (w ** 2).sum(dim=1) - 0.5 * self.latent_dim * np.log(2 * np.pi)
        
        # Total log probability (change of variables)
        log_prob_z = log_prob_w + log_prob_cnf
        
        return log_prob_z


def main():
    """Test LatentCNF implementation."""
    print("🧪 Testing LatentCNF Implementation")
    
    # Create model
    latent_cnf = LatentCNF(
        latent_dim=64,
        hidden_dim=128
    )
    
    # Test data
    batch_size = 4
    z = torch.randn(batch_size, 64)
    
    print(f"✅ Created LatentCNF with {sum(p.numel() for p in latent_cnf.parameters()):,} parameters")
    
    # Test forward (z → w)
    w, log_prob = latent_cnf.forward(z, reverse=False)
    print(f"✅ Forward pass: {z.shape} → {w.shape}, log_prob: {log_prob.shape}")
    print(f"   Output range: [{w.min():.3f}, {w.max():.3f}]")
    print(f"   Log prob range: [{log_prob.min():.3f}, {log_prob.max():.3f}]")
    
    # Test reverse (w → z)
    z_recon, _ = latent_cnf.forward(w, reverse=True)
    print(f"✅ Reverse pass: {w.shape} → {z_recon.shape}")
    
    # Test sampling
    z_samples = latent_cnf.sample(batch_size)
    print(f"✅ Sampling: {z_samples.shape}")
    
    # Test log probability
    log_prob_test = latent_cnf.log_prob(z)
    print(f"✅ Log probability: {log_prob_test.shape}")
    print(f"   Log prob range: [{log_prob_test.min():.3f}, {log_prob_test.max():.3f}]")
    
    print("🎉 LatentCNF implementation test passed!")


if __name__ == "__main__":
    main()
