"""
PointNet2D Encoder for 2D slice encoding.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class PointNet2DEncoder(nn.Module):
    """
    PointNet2D encoder with attention pooling for variable-length 2D point clouds.
    
    Based on the encoder from your CFD prediction project, adapted for VAE.
    """
    
    def __init__(self, input_dim: int = 2, latent_dim: int = 128, hidden_dim: int = 256):
        """
        Initialize the PointNet2D encoder.
        
        Args:
            input_dim: Dimension of input points (default: 2 for 2D)
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for MLP layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction MLP
        self.feature_mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, hidden_dim, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Attention mechanism for pooling
        self.attention = nn.Conv1d(hidden_dim, 1, 1)
        
        # VAE heads for mu and logvar
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input points, shape (B, N, input_dim)
            mask: Optional mask for valid points, shape (B, N)
            
        Returns:
            Tuple of (mu, logvar) for VAE latent distribution
        """
        batch_size, num_points, _ = x.shape
        
        # Validate input
        if x.shape[2] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[2]}")
        
        # Handle empty point clouds
        if num_points == 0:
            mu = torch.zeros(batch_size, self.latent_dim, device=x.device)
            logvar = torch.zeros(batch_size, self.latent_dim, device=x.device)
            return mu, logvar
        
        # Transpose for conv1d: (B, N, D) -> (B, D, N)
        x = x.transpose(1, 2)  # (B, input_dim, N)
        
        # Extract features
        features = self.feature_mlp(x)  # (B, hidden_dim, N)
        
        # Attention-based pooling
        attention_logits = self.attention(features)  # (B, 1, N)
        
        if mask is not None:
            # Apply mask to attention
            mask = mask.unsqueeze(1)  # (B, 1, N)
            attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = torch.softmax(attention_logits, dim=2)  # (B, 1, N)
        
        # Weighted sum of features
        pooled_features = torch.sum(features * attention_weights, dim=2)  # (B, hidden_dim)
        
        # VAE latent distribution parameters
        mu = self.mu_head(pooled_features)  # (B, latent_dim)
        logvar = self.logvar_head(pooled_features)  # (B, latent_dim)
        
        return mu, logvar
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for forward pass."""
        return self.forward(x, mask)
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension before the latent heads."""
        return self.hidden_dim


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick for VAE.
    
    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        
    Returns:
        Sampled latent vector
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between latent distribution and standard normal.
    
    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        
    Returns:
        KL divergence loss
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def main():
    """CLI interface for testing the PointNet2D encoder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PointNet2D Encoder")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--num-points", type=int, default=100, help="Number of points per batch")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print(f"🧪 Testing PointNet2D Encoder...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Points per batch: {args.num_points}")
    print(f"   Latent dimension: {args.latent_dim}")
    
    # Create model
    encoder = PointNet2DEncoder(latent_dim=args.latent_dim)
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test with random data
    test_points = torch.randn(args.batch_size, args.num_points, 2)
    test_mask = torch.ones(args.batch_size, args.num_points)
    
    print(f"\n🔍 Testing forward pass...")
    
    try:
        with torch.no_grad():
            mu, logvar = encoder(test_points, test_mask)
            z = reparameterize(mu, logvar)
            kl_loss = kl_divergence(mu, logvar)
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {test_points.shape}")
        print(f"   Mu shape: {mu.shape}")
        print(f"   Logvar shape: {logvar.shape}")
        print(f"   Latent sample shape: {z.shape}")
        print(f"   KL divergence: {kl_loss.mean().item():.4f}")
        
        # Test with variable lengths
        print(f"\n🔍 Testing variable lengths...")
        
        # Create batch with different point counts
        var_points = [
            torch.randn(50, 2),
            torch.randn(75, 2),
            torch.randn(30, 2),
            torch.randn(100, 2)
        ]
        
        for i, points in enumerate(var_points):
            points_batch = points.unsqueeze(0)  # Add batch dimension
            mu_var, logvar_var = encoder(points_batch)
            print(f"   Batch {i+1}: {points.shape[0]} points -> latent {mu_var.shape}")
        
        print(f"\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
