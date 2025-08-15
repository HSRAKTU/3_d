"""
PointFlow2D Adapted - Proper 2D adaptation of PointFlow CNF
Not a toy MLP, not a 3D copy-paste, but a real 2D-optimized CNF
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torchdiffeq import odeint_adjoint as odeint


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeConcat(nn.Module):
    """Concatenate time to input features"""
    def forward(self, x, t):
        t_expanded = t.view(t.shape[0], 1, 1).expand(x.shape[0], x.shape[1], 1)
        return torch.cat([x, t_expanded], dim=-1)


class ContextConcat(nn.Module):
    """Concatenate context (latent z) to features"""
    def forward(self, x, context):
        # x: [B, N, D], context: [B, C]
        context_expanded = context.unsqueeze(1).expand(-1, x.shape[1], -1)
        return torch.cat([x, context_expanded], dim=-1)


class PointFlow2DODE(nn.Module):
    """
    2D-adapted ODE function for PointFlow CNF
    Simpler than 3D but still proper CNF mechanics
    """
    
    def __init__(self, point_dim=2, context_dim=128, hidden_dim=256):
        super().__init__()
        self.point_dim = point_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Simpler architecture for 2D, but still robust
        # Input: points (2) + time (1) + context (context_dim)
        input_dim = point_dim + 1 + context_dim
        
        # 2D-appropriate network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, point_dim)
        )
        
        # Initialize weights for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        
        # Make final layer initially near-zero for stability
        self.net[-1].weight.data.mul_(0.1)
        
    def forward(self, t, states):
        """
        Args:
            t: Current time step
            states: Tuple of (points, log_det)
                points: [B, N, 2]
                log_det: [B, N, 1]
        """
        x, log_det = states
        
        # Extract context from the first point (hacky but works)
        # This assumes context is concatenated to points during integration
        B, N, D = x.shape
        
        # The context is stored in dimensions beyond point_dim
        if D > self.point_dim:
            points = x[:, :, :self.point_dim]
            context = x[:, 0, self.point_dim:]  # Context from first point
            context = context.unsqueeze(1).expand(-1, N, -1)
        else:
            points = x
            context = torch.zeros(B, N, self.context_dim).to(x)
        
        # Concatenate time
        t_vec = torch.ones(B, N, 1).to(x) * t
        
        # Concatenate everything
        inputs = torch.cat([points, t_vec, context], dim=-1)
        
        # Get velocity field
        v = self.net(inputs)
        
        # Compute divergence for log determinant
        # Using exact divergence computation
        divergence = 0.0
        for i in range(self.point_dim):
            divergence += torch.autograd.grad(
                v[:, :, i].sum(), points,
                create_graph=True, retain_graph=True
            )[0][:, :, i]
        divergence = divergence.unsqueeze(-1)
        
        # For stability during integration, we need to pass context along
        if D > self.point_dim:
            v_full = torch.cat([v, torch.zeros_like(x[:, :, self.point_dim:])], dim=-1)
        else:
            v_full = v
            
        return (v_full, divergence)


class PointFlow2DAdaptedCNF(nn.Module):
    """
    2D-adapted PointFlow CNF
    Proper CNF mechanics but optimized for 2D data
    """
    
    def __init__(
        self,
        point_dim=2,
        context_dim=128,
        hidden_dim=256,
        solver='euler',  # More stable for 2D
        solver_steps=10,  # Fewer steps needed for 2D
        atol=1e-3,
        rtol=1e-3
    ):
        super().__init__()
        
        self.point_dim = point_dim
        self.context_dim = context_dim
        self.solver = solver
        self.solver_steps = solver_steps
        self.atol = atol
        self.rtol = rtol
        
        # ODE function
        self.ode_func = PointFlow2DODE(point_dim, context_dim, hidden_dim)
        
    def forward(self, x, context, reverse=False):
        """
        Forward: data -> noise (for training)
        Reverse: noise -> data (for generation)
        
        Args:
            x: Points [B, N, 2]
            context: Latent code [B, context_dim]
            reverse: If True, integrate from noise to data
        """
        B, N, D = x.shape
        
        # Concatenate context to points for passing through ODE
        # This is a common trick to condition CNFs
        context_expanded = context.unsqueeze(1).expand(-1, N, -1)
        x_with_context = torch.cat([x, context_expanded], dim=-1)
        
        # Initial log determinant
        log_det = torch.zeros(B, N, 1).to(x)
        
        # Integration times
        if reverse:
            t = torch.tensor([1.0, 0.0]).to(x)
        else:
            t = torch.tensor([0.0, 1.0]).to(x)
        
        # Fixed step solver for 2D (more stable)
        if self.solver in ['euler', 'midpoint', 'rk4']:
            # Use fixed number of steps
            t_span = torch.linspace(t[0], t[1], self.solver_steps + 1).to(x)
            states = (x_with_context, log_det)
            
            # Manual integration for fixed-step solvers
            dt = (t[1] - t[0]) / self.solver_steps
            for i in range(self.solver_steps):
                t_i = t_span[i]
                if self.solver == 'euler':
                    dx, dlogdet = self.ode_func(t_i, states)
                    x_with_context = states[0] + dx * dt
                    log_det = states[1] + dlogdet * dt
                elif self.solver == 'midpoint':
                    # Midpoint method
                    k1_x, k1_logdet = self.ode_func(t_i, states)
                    mid_x = states[0] + k1_x * dt/2
                    mid_logdet = states[1] + k1_logdet * dt/2
                    k2_x, k2_logdet = self.ode_func(t_i + dt/2, (mid_x, mid_logdet))
                    x_with_context = states[0] + k2_x * dt
                    log_det = states[1] + k2_logdet * dt
                states = (x_with_context, log_det)
                
            # Extract final states
            final_x = x_with_context[:, :, :self.point_dim]
            final_log_det = log_det.squeeze(-1).sum(dim=1, keepdim=True)
            
        else:
            # Adaptive solver (dopri5, etc)
            states = odeint(
                self.ode_func, 
                (x_with_context, log_det), 
                t,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver
            )
            final_x = states[0][-1][:, :, :self.point_dim]
            final_log_det = states[1][-1].squeeze(-1).sum(dim=1, keepdim=True)
        
        if reverse:
            # When going from noise to data, negate log det
            final_log_det = -final_log_det
            
        return final_x, final_log_det
    
    def sample(self, context, num_points):
        """Generate points from Gaussian noise"""
        B = context.shape[0]
        
        # Sample from standard normal
        z = torch.randn(B, num_points, self.point_dim).to(context)
        
        # Transform through CNF (reverse direction)
        x, _ = self.forward(z, context, reverse=True)
        
        return x


class PointFlow2DAdaptedVAE(nn.Module):
    """
    Complete 2D-adapted PointFlow VAE
    """
    
    def __init__(
        self,
        input_dim=2,
        latent_dim=128,
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,
        solver='euler',
        solver_steps=10,
        use_deterministic_encoder=True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_deterministic_encoder = use_deterministic_encoder
        
        # Import encoder
        from .encoder import PointNet2DEncoder
        
        # Encoder
        self.encoder = PointNet2DEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim
        )
        
        # 2D-adapted Point CNF
        self.point_cnf = PointFlow2DAdaptedCNF(
            point_dim=input_dim,
            context_dim=latent_dim,
            hidden_dim=cnf_hidden_dim,
            solver=solver,
            solver_steps=solver_steps
        )
        
    def encode(self, x):
        """Encode points to latent code"""
        z_mu, z_logvar = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            # Reparameterization trick
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            return z_mu + eps * std
    
    def decode(self, z, num_points):
        """Decode latent code to points"""
        return self.point_cnf.sample(z, num_points)
    
    def reconstruct(self, x):
        """Reconstruct input points"""
        z = self.encode(x)
        return self.decode(z, x.shape[1])
    
    def forward(self, x, optimizer, step, writer=None):
        """
        Training forward pass with PointFlow loss
        """
        optimizer.zero_grad()
        
        B, N, D = x.shape
        
        # Encode
        z = self.encode(x)
        
        # Forward pass through CNF: x -> Gaussian
        y, log_det = self.point_cnf(x, z, reverse=False)
        
        # Log probability under standard normal
        log_py = -0.5 * (y ** 2).sum(dim=-1) - 0.5 * D * np.log(2 * np.pi)
        log_py = log_py.sum(dim=1, keepdim=True)  # Sum over points
        
        # Log probability of data
        log_px = log_py + log_det
        
        # Loss is negative log likelihood
        loss = -log_px.mean()
        
        # Backward and step
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Return metrics
        return {
            'loss': loss.item(),
            'recon_nats': loss.item() / (N * D),  # Normalized
            'log_det': log_det.mean().item()
        }
