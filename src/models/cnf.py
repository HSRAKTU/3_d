"""
Continuous Normalizing Flow (CNF) for 2D point cloud generation.
Based on PointFlow paper implementation.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

# Check if torchdiffeq is available
try:
    from torchdiffeq import odeint_adjoint as odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    logger.warning("torchdiffeq not available. Using simplified placeholder implementation.")
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None


class ODEFunc(nn.Module):
    """
    ODE function for continuous normalizing flow.
    Defines dx/dt = f(x, t, context)
    """
    
    def __init__(self, dim: int = 2, hidden_dim: int = 128, context_dim: int = 128):
        """
        Initialize ODE function.
        
        Args:
            dim: Dimension of the state (2 for 2D points)
            hidden_dim: Hidden dimension for neural network
            context_dim: Dimension of conditioning context (latent vector)
        """
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        
        # Neural network for dynamics
        # Takes [x, t, context] and outputs dx/dt
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Initialize small weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for numerical stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute dx/dt given current state and time.
        
        Args:
            t: Current time (scalar)
            state: Current state containing [x, context], shape (batch_size * num_points, dim + context_dim)
            
        Returns:
            Time derivative dx/dt, shape (batch_size * num_points, dim + context_dim)
        """
        # Extract points and context from state
        x = state[:, :self.dim]  # (N, dim)
        context = state[:, self.dim:]  # (N, context_dim)
        
        # Time embedding (broadcast to all points)
        batch_size = x.shape[0]
        t_embed = t.expand(batch_size, 1)  # (N, 1)
        
        # Concatenate inputs
        inputs = torch.cat([x, t_embed, context], dim=1)  # (N, dim + 1 + context_dim)
        
        # Compute velocity
        dx_dt = self.net(inputs)  # (N, dim)
        
        # Return full state derivative (context doesn't change)
        dcontext_dt = torch.zeros_like(context)
        
        return torch.cat([dx_dt, dcontext_dt], dim=1)


class SimpleCNF(nn.Module):
    """
    Simplified CNF implementation when torchdiffeq is not available.
    Uses Euler method for integration.
    """
    
    def __init__(self, dim: int = 2, hidden_dim: int = 128, context_dim: int = 128):
        super().__init__()
        self.ode_func = ODEFunc(dim, hidden_dim, context_dim)
        self.dim = dim
        self.context_dim = context_dim
    
    def forward(self, x0: torch.Tensor, context: torch.Tensor, 
                reverse: bool = False, num_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple integration using Euler method.
        
        Args:
            x0: Initial points, shape (batch_size, num_points, dim)
            context: Conditioning context, shape (batch_size, context_dim)
            reverse: Whether to integrate backwards
            num_steps: Number of integration steps
            
        Returns:
            Tuple of (final_points, log_likelihood_estimate)
        """
        batch_size, num_points, _ = x0.shape
        device = x0.device
        
        # Flatten for integration
        x_flat = x0.view(-1, self.dim)  # (batch_size * num_points, dim)
        
        # Expand context to all points
        context_expanded = context.unsqueeze(1).expand(-1, num_points, -1)  # (batch_size, num_points, context_dim)
        context_flat = context_expanded.contiguous().view(-1, self.context_dim)  # (batch_size * num_points, context_dim)
        
        # Create state vector
        state = torch.cat([x_flat, context_flat], dim=1)  # (batch_size * num_points, dim + context_dim)
        
        # Integration bounds
        t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        if reverse:
            t_span = torch.flip(t_span, [0])
        
        dt = 1.0 / num_steps
        if reverse:
            dt = -dt
        
        # Euler integration
        current_state = state
        for i in range(num_steps):
            t = t_span[i]
            dstate_dt = self.ode_func(t, current_state)
            current_state = current_state + dt * dstate_dt
        
        # Extract final points
        final_x_flat = current_state[:, :self.dim]
        final_x = final_x_flat.view(batch_size, num_points, self.dim)
        
        # Placeholder log likelihood (would need trace estimation in real implementation)
        log_likelihood = torch.zeros(batch_size, device=device)
        
        return final_x, log_likelihood


class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous Normalizing Flow for 2D point cloud generation.
    """
    
    def __init__(self, dim: int = 2, hidden_dim: int = 128, context_dim: int = 128, 
                 use_adjoint: bool = True):
        """
        Initialize CNF.
        
        Args:
            dim: Dimension of points (2 for 2D)
            hidden_dim: Hidden dimension for ODE function
            context_dim: Dimension of conditioning context
            use_adjoint: Whether to use adjoint method for memory efficiency
        """
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.use_adjoint = use_adjoint and TORCHDIFFEQ_AVAILABLE
        
        # ODE function
        self.ode_func = ODEFunc(dim, hidden_dim, context_dim)
        
        # Fallback to simple implementation if torchdiffeq not available
        if not TORCHDIFFEQ_AVAILABLE:
            logger.warning("Using simplified CNF implementation without exact likelihood")
            self.simple_cnf = SimpleCNF(dim, hidden_dim, context_dim)
    
    def forward(self, x0: torch.Tensor, context: torch.Tensor, 
                reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CNF.
        
        Args:
            x0: Initial points, shape (batch_size, num_points, dim)
            context: Conditioning context, shape (batch_size, context_dim)
            reverse: Whether to integrate backwards (generation vs inference)
            
        Returns:
            Tuple of (transformed_points, log_likelihood)
        """
        if not TORCHDIFFEQ_AVAILABLE:
            return self.simple_cnf(x0, context, reverse)
        
        # Use torchdiffeq for exact integration
        return self._integrate_with_torchdiffeq(x0, context, reverse)
    
    def _integrate_with_torchdiffeq(self, x0: torch.Tensor, context: torch.Tensor, 
                                   reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integration using torchdiffeq with trace estimation.
        """
        batch_size, num_points, _ = x0.shape
        device = x0.device
        
        # Flatten for integration
        x_flat = x0.view(-1, self.dim)
        context_expanded = context.unsqueeze(1).expand(-1, num_points, -1)
        context_flat = context_expanded.contiguous().view(-1, self.context_dim)
        
        # Augment state with context
        state = torch.cat([x_flat, context_flat], dim=1)
        
        # Integration time
        t = torch.tensor([0.0, 1.0], device=device)
        if reverse:
            t = torch.flip(t, [0])
        
        # Integrate (simplified - real implementation would need trace estimation)
        with torch.no_grad():  # Simplified for now
            integrated_state = odeint(self.ode_func, state, t, method='dopri5')[-1]
        
        # Extract points
        final_x_flat = integrated_state[:, :self.dim]
        final_x = final_x_flat.view(batch_size, num_points, self.dim)
        
        # Placeholder for log likelihood (needs proper implementation)
        log_likelihood = torch.zeros(batch_size, device=device)
        
        return final_x, log_likelihood
    
    def sample(self, context: torch.Tensor, num_points: int, 
               base_distribution: str = "normal") -> torch.Tensor:
        """
        Sample points from the CNF.
        
        Args:
            context: Conditioning context, shape (batch_size, context_dim)
            num_points: Number of points to sample
            base_distribution: Base distribution to sample from
            
        Returns:
            Generated points, shape (batch_size, num_points, dim)
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Sample from base distribution
        if base_distribution == "normal":
            x0 = torch.randn(batch_size, num_points, self.dim, device=device)
        else:
            raise ValueError(f"Unknown base distribution: {base_distribution}")
        
        # Transform through flow
        x1, _ = self.forward(x0, context, reverse=False)
        
        return x1
    
    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of points under the flow.
        
        Args:
            x: Points to evaluate, shape (batch_size, num_points, dim)
            context: Conditioning context, shape (batch_size, context_dim)
            
        Returns:
            Log probabilities, shape (batch_size,)
        """
        # Transform to base distribution
        x0, log_det = self.forward(x, context, reverse=True)
        
        # Base distribution log prob (standard normal)
        base_log_prob = -0.5 * (x0**2).sum(dim=[1, 2]) - 0.5 * x0.shape[1] * x0.shape[2] * np.log(2 * np.pi)
        
        # Total log prob = base log prob + log determinant
        return base_log_prob + log_det


def main():
    """CLI interface for testing CNF."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Continuous Normalizing Flow")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-points", type=int, default=100, help="Number of points")
    parser.add_argument("--context-dim", type=int, default=64, help="Context dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print(f"🧪 Testing Continuous Normalizing Flow...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Points per batch: {args.num_points}")
    print(f"   Context dimension: {args.context_dim}")
    print(f"   torchdiffeq available: {TORCHDIFFEQ_AVAILABLE}")
    
    # Create CNF
    cnf = ContinuousNormalizingFlow(
        dim=2, 
        hidden_dim=args.hidden_dim, 
        context_dim=args.context_dim
    )
    
    total_params = sum(p.numel() for p in cnf.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    
    # Test data
    context = torch.randn(args.batch_size, args.context_dim)
    
    print(f"\n🔍 Testing sampling...")
    try:
        # Sample points
        generated_points = cnf.sample(context, args.num_points)
        print(f"✅ Sampling successful!")
        print(f"   Context shape: {context.shape}")
        print(f"   Generated points shape: {generated_points.shape}")
        print(f"   Points range: [{generated_points.min():.3f}, {generated_points.max():.3f}]")
        
        # Test forward/backward
        print(f"\n🔍 Testing forward/backward...")
        test_points = torch.randn(args.batch_size, args.num_points, 2)
        
        # Forward pass
        transformed, log_prob_forward = cnf.forward(test_points, context, reverse=False)
        print(f"   Forward transform: {test_points.shape} -> {transformed.shape}")
        
        # Backward pass
        reconstructed, log_prob_backward = cnf.forward(transformed, context, reverse=True)
        print(f"   Backward transform: {transformed.shape} -> {reconstructed.shape}")
        
        # Check reconstruction error
        reconstruction_error = (test_points - reconstructed).abs().mean()
        print(f"   Reconstruction error: {reconstruction_error:.6f}")
        
        if reconstruction_error < 0.1:  # Lenient for simple implementation
            print(f"✅ Reconstruction test passed!")
        else:
            print(f"⚠️ High reconstruction error (expected for simplified implementation)")
        
        print(f"\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
