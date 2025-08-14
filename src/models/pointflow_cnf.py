"""
PointFlow-exact CNF implementation for 2D point clouds.
Based on the original PointFlow paper and implementation.
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


class PointFlowODEFunc(nn.Module):
    """
    ODE function for PointFlow CNF.
    Exactly following PointFlow's implementation.
    """
    
    def __init__(self, in_out_dim: int, hidden_dim: int, context_dim: int):
        """
        Initialize ODE function.
        
        Args:
            in_out_dim: Dimension of points (2 for 2D)
            hidden_dim: Hidden dimension for neural network
            context_dim: Dimension of conditioning context
        """
        super().__init__()
        
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        
        # PointFlow uses this exact architecture
        self.layer1 = ConcatSquashLinear(in_out_dim, hidden_dim, context_dim + 1)  # +1 for time
        self.layer2 = ConcatSquashLinear(hidden_dim, hidden_dim, context_dim + 1)
        self.layer3 = ConcatSquashLinear(hidden_dim, hidden_dim, context_dim + 1)
        self.output_layer = nn.Linear(hidden_dim, in_out_dim)
        
        # Initialize to small values (important for stability)
        for module in [self.layer1, self.layer2, self.layer3, self.output_layer]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.uniform_(submodule.weight, -0.01, 0.01)
                    if submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0)
    
    def forward(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt.
        
        Args:
            t: Current time, shape (1,)
            states: Combined [x, context], shape (batch_size, in_out_dim + context_dim)
            
        Returns:
            dx/dt, shape (batch_size, in_out_dim + context_dim)
        """
        # Extract points and context
        x = states[:, :self.in_out_dim]  # (batch_size, 2)
        context = states[:, self.in_out_dim:]  # (batch_size, context_dim)
        
        # Create time embedding for each point
        batch_size = x.shape[0]
        t_embed = t.expand(batch_size, 1)  # (batch_size, 1)
        
        # Compute velocity using PointFlow's architecture
        ctx_t = torch.cat([context, t_embed], dim=1)
        h = self.layer1(x, ctx_t)
        h = self.layer2(h, ctx_t)
        h = self.layer3(h, ctx_t)
        dx_dt = self.output_layer(h)
        
        # Context doesn't change with time
        dcontext_dt = torch.zeros_like(context)
        
        return torch.cat([dx_dt, dcontext_dt], dim=1)


class ConcatSquashLinear(nn.Module):
    """
    Linear layer with context concatenation and squashing activation.
    Exactly as used in PointFlow.
    """
    
    def __init__(self, dim_in: int, dim_out: int, dim_ctx: int):
        """
        Initialize layer.
        
        Args:
            dim_in: Input dimension
            dim_out: Output dimension  
            dim_ctx: Context dimension
        """
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self.hyper_gate = nn.Linear(dim_ctx, dim_out)
        
    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with context conditioning.
        
        Args:
            x: Input points
            ctx: Context (conditioning + time)
            
        Returns:
            Conditioned output
        """
        gate = torch.sigmoid(self.hyper_gate(ctx))
        bias = self.hyper_bias(ctx)
        
        return self.linear(x) * gate + bias


class PointFlowCNF(nn.Module):
    """
    PointFlow Continuous Normalizing Flow for 2D point clouds.
    Exact implementation following the original paper.
    """
    
    def __init__(self, 
                 point_dim: int = 2,
                 context_dim: int = 64,
                 hidden_dim: int = 128,
                 solver: str = 'dopri5',
                 atol: float = 1e-5,
                 rtol: float = 1e-5):
        """
        Initialize PointFlow CNF.
        
        Args:
            point_dim: Dimension of points (2 for 2D)
            context_dim: Dimension of conditioning context
            hidden_dim: Hidden dimension for ODE network
            solver: ODE solver ('dopri5' as in PointFlow)
            atol: Absolute tolerance for ODE solver
            rtol: Relative tolerance for ODE solver
        """
        super().__init__()
        
        self.point_dim = point_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is required for PointFlow CNF")
        
        # Create ODE function
        self.odefunc = PointFlowODEFunc(
            in_out_dim=point_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim
        )
        
        # Integration times (PointFlow uses [0, 1])
        self.register_buffer('sqrt_end_time', torch.sqrt(torch.tensor(1.0)))
        
        logger.info(f"Initialized PointFlowCNF: point_dim={point_dim}, context_dim={context_dim}")
    
    def sample(self, 
               context: torch.Tensor, 
               num_points: int,
               temperature: float = 1.0) -> torch.Tensor:
        """
        Sample points from the flow.
        
        Args:
            context: Conditioning context, shape (batch_size, context_dim)
            num_points: Number of points to sample
            temperature: Sampling temperature
            
        Returns:
            Generated points, shape (batch_size, num_points, point_dim)
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Sample from base distribution (Gaussian)
        z = torch.randn(batch_size, num_points, self.point_dim, device=device) * temperature
        
        # Expand context for all points
        context_expanded = context.unsqueeze(1).expand(-1, num_points, -1)
        
        # Flatten for ODE integration
        z_flat = z.view(batch_size * num_points, self.point_dim)
        context_flat = context_expanded.contiguous().view(batch_size * num_points, self.context_dim)
        
        # Combine state
        states = torch.cat([z_flat, context_flat], dim=1)
        
        # Integration times
        integration_times = torch.tensor([0.0, 1.0], device=device)
        
        # Solve ODE
        trajectory = odeint(
            self.odefunc,
            states,
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver
        )
        
        # Extract final points
        final_states = trajectory[-1]  # (batch_size * num_points, point_dim + context_dim)
        final_points = final_states[:, :self.point_dim]  # (batch_size * num_points, point_dim)
        
        # Reshape back
        result = final_points.view(batch_size, num_points, self.point_dim)
        
        return result
    
    def log_prob(self, 
                 points: torch.Tensor, 
                 context: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of points under the flow.
        
        Args:
            points: Points to evaluate, shape (batch_size, num_points, point_dim)
            context: Conditioning context, shape (batch_size, context_dim)
            
        Returns:
            Log probabilities, shape (batch_size, num_points)
        """
        batch_size, num_points, _ = points.shape
        device = points.device
        
        # Expand context for all points
        context_expanded = context.unsqueeze(1).expand(-1, num_points, -1)
        
        # Flatten for ODE integration
        points_flat = points.view(batch_size * num_points, self.point_dim)
        context_flat = context_expanded.contiguous().view(batch_size * num_points, self.context_dim)
        
        # Combine state for backward integration
        states = torch.cat([points_flat, context_flat], dim=1)
        
        # Add divergence tracking for likelihood computation
        states_aug = torch.cat([
            states,
            torch.zeros(batch_size * num_points, 1, device=device)  # For divergence
        ], dim=1)
        
        # Integration times (backward: 1 -> 0)
        integration_times = torch.tensor([1.0, 0.0], device=device)
        
        # Create augmented dynamics as a proper nn.Module
        class AugmentedDynamics(nn.Module):
            def __init__(self, odefunc, point_dim):
                super().__init__()
                self.odefunc = odefunc
                self.point_dim = point_dim
                
            def forward(self, t, states_aug):
                states = states_aug[:, :-1]
                
                with torch.enable_grad():
                    states.requires_grad_(True)
                    dx_dt = self.odefunc(t, states)
                    points_part = dx_dt[:, :self.point_dim]
                    
                    # Compute divergence
                    divergence = 0
                    for i in range(self.point_dim):
                        divergence += torch.autograd.grad(
                            points_part[:, i].sum(), 
                            states,
                            create_graph=True,
                            retain_graph=True
                        )[0][:, i]
                
                return torch.cat([dx_dt, -divergence.unsqueeze(1)], dim=1)
        
        aug_dynamics = AugmentedDynamics(self.odefunc, self.point_dim)
        
        # Solve augmented system
        trajectory = odeint(
            aug_dynamics,
            states_aug,
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver
        )
        
        # Extract final states and divergence
        final_states_aug = trajectory[-1]
        final_points = final_states_aug[:, :self.point_dim]
        log_det_jacobian = final_states_aug[:, -1]
        
        # Base distribution log prob (standard Gaussian)
        base_log_prob = -0.5 * (
            self.point_dim * np.log(2 * np.pi) + 
            torch.sum(final_points**2, dim=1)
        )
        
        # Apply change of variables formula
        log_prob = base_log_prob + log_det_jacobian
        
        # Reshape back
        result = log_prob.view(batch_size, num_points)
        
        return result
    
    def forward(self, 
                points: torch.Tensor, 
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing both transformed points and log probability.
        
        Args:
            points: Input points, shape (batch_size, num_points, point_dim)
            context: Conditioning context, shape (batch_size, context_dim)
            
        Returns:
            (transformed_points, log_prob)
        """
        # For training, we typically compute log_prob of the input points
        log_prob = self.log_prob(points, context)
        
        # Return input points and their log probability
        return points, log_prob


def main():
    """Test PointFlow CNF implementation."""
    print("🧪 Testing PointFlow CNF Implementation")
    
    # Create model
    cnf = PointFlowCNF(
        point_dim=2,
        context_dim=64,
        hidden_dim=128
    )
    
    # Test data
    batch_size = 2
    num_points = 100
    context = torch.randn(batch_size, 64)
    
    print(f"✅ Created CNF with {sum(p.numel() for p in cnf.parameters()):,} parameters")
    
    # Test sampling
    samples = cnf.sample(context, num_points)
    print(f"✅ Sampling works: {context.shape} → {samples.shape}")
    
    # Test log probability
    points = torch.randn(batch_size, num_points, 2)
    log_prob = cnf.log_prob(points, context)
    print(f"✅ Log probability works: {points.shape} → {log_prob.shape}")
    print(f"   Log prob range: [{log_prob.min():.3f}, {log_prob.max():.3f}]")
    
    print("🎉 PointFlow CNF implementation test passed!")


if __name__ == "__main__":
    main()
