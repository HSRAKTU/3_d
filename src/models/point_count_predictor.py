"""
Point Count Predictor MLP
Predicts the number of points a latent code should generate.
"""

import torch
import torch.nn as nn
import numpy as np


class PointCountPredictor(nn.Module):
    """
    MLP that predicts the number of points for a given latent code.
    This allows the model to learn the relationship between shape complexity
    and point density.
    """
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        """
        Initialize the point count predictor.
        
        Args:
            latent_dim: Dimension of latent codes
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),  # Output single value
            nn.Softplus()  # Ensure positive output
        )
        
        # Scale factor to map to reasonable point counts
        # Will be learned during training
        self.scale = nn.Parameter(torch.tensor(1000.0))
        self.offset = nn.Parameter(torch.tensor(100.0))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict number of points from latent code.
        
        Args:
            z: Latent codes, shape (batch_size, latent_dim)
            
        Returns:
            Predicted point counts, shape (batch_size,)
        """
        raw_output = self.network(z).squeeze(-1)
        # Scale to reasonable range (e.g., 100-5000 points)
        point_counts = raw_output * self.scale + self.offset
        return point_counts
    
    def get_discrete_counts(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get discrete (integer) point counts.
        
        Args:
            z: Latent codes
            
        Returns:
            Integer point counts
        """
        continuous_counts = self.forward(z)
        # Round to nearest integer
        discrete_counts = torch.round(continuous_counts).long()
        # Ensure minimum count
        discrete_counts = torch.clamp(discrete_counts, min=50)
        return discrete_counts


class PointFlow2DVAEWithCountPredictor(nn.Module):
    """
    Extended PointFlow2D VAE that includes point count prediction.
    """
    
    def __init__(self, base_vae, point_count_predictor):
        """
        Wrap existing VAE with point count predictor.
        
        Args:
            base_vae: The base PointFlow2DVAE model
            point_count_predictor: The point count predictor
        """
        super().__init__()
        self.vae = base_vae
        self.count_predictor = point_count_predictor
    
    def forward(self, x, opt, step, writer=None):
        """Forward pass with point count loss."""
        # Get base VAE losses
        result = self.vae.forward(x, opt, step, writer)
        
        # Additionally train count predictor
        with torch.no_grad():
            z = self.vae.encode(x)
        
        # Predict point counts
        predicted_counts = self.count_predictor(z)
        actual_counts = x.shape[1] * torch.ones(x.shape[0]).to(x.device)
        
        # Add count prediction loss
        count_loss = nn.functional.mse_loss(predicted_counts, actual_counts)
        result['count_loss'] = count_loss.item()
        
        # Backprop for count predictor
        count_loss.backward()
        
        return result
    
    def sample_with_adaptive_counts(self, batch_size, truncate_std=None, gpu=None):
        """
        Sample with automatically determined point counts.
        
        Args:
            batch_size: Number of samples to generate
            truncate_std: Optional truncation for sampling
            gpu: GPU device
            
        Returns:
            List of (z, x) tuples with varying point counts
        """
        # First, sample latent codes
        device = next(self.vae.parameters()).device if gpu is None else f'cuda:{gpu}'
        
        # Generate shape codes
        w = torch.randn(batch_size, self.vae.latent_dim).to(device)
        if truncate_std is not None:
            w = torch.clamp(w, -truncate_std, truncate_std)
        
        z, _ = self.vae.latent_cnf(w, None, reverse=True)
        z = z.view(*w.size())
        
        # Predict point counts for each latent code
        point_counts = self.count_predictor.get_discrete_counts(z)
        
        # Generate samples with adaptive point counts
        samples = []
        for i in range(batch_size):
            num_points = point_counts[i].item()
            
            # Generate points for this specific count
            y = torch.randn(1, num_points, self.vae.input_dim).to(device)
            if truncate_std is not None:
                y = torch.clamp(y, -truncate_std, truncate_std)
            
            x, _ = self.vae.point_cnf(y, z[i:i+1], reverse=True)
            x = x.view(num_points, self.vae.input_dim)
            
            samples.append({
                'latent_code': z[i],
                'points': x,
                'num_points': num_points,
                'predicted_count': point_counts[i].item()
            })
        
        return samples


def visualize_point_count_learning(model, test_loader, device, save_path):
    """
    Visualize how well the model learned point count prediction.
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    actual_counts = []
    predicted_counts = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['points'].to(device)
            z = model.vae.encode(x)
            
            pred_counts = model.count_predictor(z).cpu().numpy()
            actual = batch['num_points'].numpy()
            
            predicted_counts.extend(pred_counts)
            actual_counts.extend(actual)
            
            if len(actual_counts) > 100:  # Sample for visualization
                break
    
    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_counts, predicted_counts, alpha=0.6)
    
    # Add diagonal line
    min_val = min(min(actual_counts), min(predicted_counts))
    max_val = max(max(actual_counts), max(predicted_counts))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    plt.xlabel('Actual Point Count')
    plt.ylabel('Predicted Point Count')
    plt.title('Point Count Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Compute metrics
    actual = np.array(actual_counts)
    predicted = np.array(predicted_counts)
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    
    print(f"Point Count Prediction Metrics:")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  Mean Actual: {actual.mean():.1f} ± {actual.std():.1f}")
    print(f"  Mean Predicted: {predicted.mean():.1f} ± {predicted.std():.1f}")


if __name__ == "__main__":
    # Example usage
    print("Point Count Predictor Module")
    
    # Test the predictor
    predictor = PointCountPredictor(latent_dim=64, hidden_dim=128)
    
    # Test with random latent codes
    z = torch.randn(5, 64)
    counts = predictor(z)
    discrete_counts = predictor.get_discrete_counts(z)
    
    print(f"Continuous counts: {counts}")
    print(f"Discrete counts: {discrete_counts}")
