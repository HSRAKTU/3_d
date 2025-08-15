"""
Lightweight PointFlow CNF specifically designed for 2D.
Much simpler than the 3D version - appropriate for our use case.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint

logger = logging.getLogger(__name__)


class Simple2DODE(nn.Module):
    """
    MUCH simpler ODE for 2D point clouds.
    No fancy ConcatSquash, just basic MLPs.
    """
    
    def __init__(self, point_dim: int = 2, context_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.point_dim = point_dim
        self.context_dim = context_dim
        
        # Simple architecture: just 2 layers!
        self.net = nn.Sequential(
            nn.Linear(point_dim + context_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),  # Bounded activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, point_dim)
        )
        
        # Initialize small
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, states):
        """Simple forward: just concat everything and pass through MLP."""
        # Extract points and context
        x = states[:, :self.point_dim]
        ctx = states[:, self.point_dim:]
        
        # Expand time
        t_vec = t.expand(x.shape[0], 1)
        
        # Simple concat and forward
        inputs = torch.cat([x, ctx, t_vec], dim=1)
        dx_dt = self.net(inputs)
        
        # Context doesn't change
        dctx_dt = torch.zeros_like(ctx)
        
        return torch.cat([dx_dt, dctx_dt], dim=1)


class PointFlow2DCNF(nn.Module):
    """
    Lightweight CNF designed specifically for 2D automotive slices.
    Much simpler than the original 3D PointFlow.
    """
    
    def __init__(self,
                 point_dim: int = 2,
                 context_dim: int = 128,
                 hidden_dim: int = 64,  # Much smaller!
                 solver: str = 'euler',  # Simpler solver
                 solver_steps: int = 10,  # Fixed steps instead of adaptive
                 ):
        super().__init__()
        
        self.point_dim = point_dim
        self.context_dim = context_dim
        
        # Simple ODE
        self.odefunc = Simple2DODE(point_dim, context_dim, hidden_dim)
        
        # For fixed-step solvers
        self.solver = solver
        self.solver_steps = solver_steps
        
        logger.info(f"Created lightweight 2D CNF with {sum(p.numel() for p in self.parameters())} params")
    
    def sample(self, context: torch.Tensor, num_points: int, temperature: float = 1.0) -> torch.Tensor:
        """Generate points from latent code."""
        batch_size = context.shape[0]
        device = context.device
        
        # Start from Gaussian
        z = torch.randn(batch_size, num_points, self.point_dim, device=device) * temperature
        
        # Expand context
        ctx_exp = context.unsqueeze(1).expand(-1, num_points, -1)
        
        # Flatten
        z_flat = z.view(-1, self.point_dim)
        ctx_flat = ctx_exp.reshape(-1, self.context_dim)
        
        # Combine
        states = torch.cat([z_flat, ctx_flat], dim=1)
        
        # Simple integration with fixed steps
        if self.solver == 'euler':
            # Manual Euler integration (very simple!)
            dt = 1.0 / self.solver_steps
            t = torch.tensor(0.0, device=device)
            
            for _ in range(self.solver_steps):
                states = states + dt * self.odefunc(t, states)
                t = t + dt
        else:
            # Use torchdiffeq
            times = torch.linspace(0, 1, 2, device=device)
            states = odeint(self.odefunc, states, times, method=self.solver)[-1]
        
        # Extract points
        points = states[:, :self.point_dim]
        return points.view(batch_size, num_points, self.point_dim)
    
    def forward(self, points, context=None, reverse=True, **kwargs):
        """Compatible interface with original PointFlowCNF."""
        if reverse:
            # Decode mode
            batch_size, num_points = points.shape[:2]
            if context is None:
                context = torch.zeros(batch_size, self.context_dim, device=points.device)
            return self.sample(context, num_points), torch.zeros(batch_size, num_points, 1).to(points)
        else:
            # Forward mode - transform real points to Gaussian
            batch_size, num_points = points.shape[:2]
            if context is None:
                context = torch.zeros(batch_size, self.context_dim, device=points.device)
            
            # Expand context 
            ctx_exp = context.unsqueeze(1).expand(-1, num_points, -1)
            
            # Flatten
            points_flat = points.view(-1, self.point_dim)
            ctx_flat = ctx_exp.reshape(-1, self.context_dim)
            
            # Combine
            states = torch.cat([points_flat, ctx_flat], dim=1)
            
            # Reverse integration (points -> Gaussian)
            if self.solver == 'euler':
                dt = -1.0 / self.solver_steps  # Negative for reverse
                t = torch.tensor(1.0, device=points.device)
                
                for _ in range(self.solver_steps):
                    states = states + dt * self.odefunc(t, states)
                    t = t + dt
            else:
                times = torch.linspace(1, 0, 2, device=points.device)  # Reverse time
                states = odeint(self.odefunc, states, times, method=self.solver)[-1]
            
            # Extract transformed points (should be Gaussian-like)
            gaussian_points = states[:, :self.point_dim]
            gaussian_points = gaussian_points.view(batch_size, num_points, self.point_dim)
            
            # Return transformed points and dummy log determinant
            delta_log_p = torch.zeros(batch_size, num_points, 1, device=points.device)
            return gaussian_points, delta_log_p


def test_lightweight():
    """Quick test of lightweight CNF."""
    print("Testing Lightweight 2D CNF...")
    
    # Create model
    cnf = PointFlow2DCNF(hidden_dim=32, solver='euler', solver_steps=5)
    print(f"Parameters: {sum(p.numel() for p in cnf.parameters())}")
    
    # Test generation
    z = torch.randn(2, 64)
    points = cnf.sample(z, 100)
    print(f"Generated shape: {points.shape}")
    print("✓ Lightweight CNF works!")


if __name__ == "__main__":
    test_lightweight()
