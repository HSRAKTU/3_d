"""
Visualization and analysis tools for PointFlow2D.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.pointflow2d import PointFlow2DVAE
from training.dataset import SliceDataset
from data.preprocessor import SlicePreprocessor

logger = logging.getLogger(__name__)


class SliceVisualizer:
    """
    Comprehensive visualization tools for PointFlow2D analysis.
    """
    
    def __init__(self, model: PointFlow2DVAE, device: str = "auto"):
        """
        Initialize visualizer.
        
        Args:
            model: Trained PointFlow2D model
            device: Device to run inference on
        """
        self.model = model
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        self.preprocessor = SlicePreprocessor()
    
    def visualize_reconstruction(self, 
                               input_points: torch.Tensor,
                               mask: Optional[torch.Tensor] = None,
                               num_samples: int = 3,
                               figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Visualize input vs reconstruction vs multiple samples.
        
        Args:
            input_points: Input points, shape (1, N, 2) or (N, 2)
            mask: Optional mask for valid points
            num_samples: Number of reconstruction samples to show
            figsize: Figure size
        """
        if input_points.dim() == 2:
            input_points = input_points.unsqueeze(0)
        
        if mask is None:
            mask = torch.ones(input_points.shape[:2])
        elif mask.dim() == 1:
            mask = mask.unsqueeze(0)
        
        input_points = input_points.to(self.device)
        mask = mask.to(self.device)
        
        with torch.no_grad():
            # Get reconstruction and latent
            result = self.model.forward(input_points, mask)
            z = result['z']
            
            # Generate multiple samples from same latent
            reconstructions = []
            for _ in range(num_samples):
                recon = self.model.decode(z, input_points.shape[1])
                reconstructions.append(recon[0].cpu().numpy())
        
        # Plot
        fig, axes = plt.subplots(1, num_samples + 1, figsize=figsize)
        
        # Original
        valid_points = input_points[0][mask[0].bool()].cpu().numpy()
        axes[0].scatter(valid_points[:, 0], valid_points[:, 1], s=2, alpha=0.7, c='blue')
        axes[0].set_title('Original Slice')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        
        # Reconstructions
        for i, recon in enumerate(reconstructions):
            axes[i + 1].scatter(recon[:, 0], recon[:, 1], s=2, alpha=0.7, c='red')
            axes[i + 1].set_title(f'Reconstruction {i + 1}')
            axes[i + 1].set_aspect('equal')
            axes[i + 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        kl_loss = result['kl_loss'].item()
        print(f"📊 Reconstruction Metrics:")
        print(f"   KL Divergence: {kl_loss:.4f}")
        print(f"   Latent norm: {torch.norm(z).item():.4f}")
    
    def compare_slices(self, 
                      slice1: torch.Tensor, 
                      slice2: torch.Tensor,
                      labels: Tuple[str, str] = ("Slice 1", "Slice 2"),
                      figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Compare two slices side by side.
        
        Args:
            slice1: First slice points
            slice2: Second slice points  
            labels: Labels for the slices
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot slice 1
        if isinstance(slice1, torch.Tensor):
            slice1 = slice1.cpu().numpy()
        axes[0].scatter(slice1[:, 0], slice1[:, 1], s=2, alpha=0.7, c='blue')
        axes[0].set_title(labels[0])
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        
        # Plot slice 2
        if isinstance(slice2, torch.Tensor):
            slice2 = slice2.cpu().numpy()
        axes[1].scatter(slice2[:, 0], slice2[:, 1], s=2, alpha=0.7, c='red')
        axes[1].set_title(labels[1])
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_samples(self, 
                        num_samples: int = 9,
                        num_points: int = 500,
                        figsize: Tuple[int, int] = (15, 15)) -> List[np.ndarray]:
        """
        Generate random samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            num_points: Number of points per sample
            figsize: Figure size
            
        Returns:
            List of generated point clouds
        """
        with torch.no_grad():
            samples = self.model.sample(num_samples, num_points, self.device)
            samples = samples.cpu().numpy()
        
        # Plot in grid
        cols = int(np.ceil(np.sqrt(num_samples)))
        rows = int(np.ceil(num_samples / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            row, col = i // cols, i % cols
            axes[row, col].scatter(samples[i, :, 0], samples[i, :, 1], s=1, alpha=0.7)
            axes[row, col].set_title(f'Sample {i + 1}')
            axes[row, col].set_aspect('equal')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return samples
    
    def analyze_latent_space(self, 
                            dataset: SliceDataset,
                            num_samples: int = 200,
                            figsize: Tuple[int, int] = (12, 8)) -> np.ndarray:
        """
        Analyze and visualize the latent space.
        
        Args:
            dataset: Dataset to analyze
            num_samples: Number of samples to analyze
            figsize: Figure size
            
        Returns:
            Latent embeddings array
        """
        embeddings = []
        car_ids = []
        slice_indices = []
        
        # Sample data points
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        with torch.no_grad():
            for idx in indices:
                item = dataset[idx]
                points = item['points'].unsqueeze(0).to(self.device)
                
                mu, logvar = self.model.encode(points)
                embeddings.append(mu[0].cpu().numpy())
                car_ids.append(item['car_id'])
                slice_indices.append(item['slice_idx'])
        
        embeddings = np.array(embeddings)
        
        # Visualize first 2 dimensions
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Color by car ID
        unique_cars = list(set(car_ids))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cars)))
        car_to_color = {car: colors[i] for i, car in enumerate(unique_cars)}
        
        for i, car_id in enumerate(car_ids):
            axes[0].scatter(embeddings[i, 0], embeddings[i, 1], 
                          c=[car_to_color[car_id]], s=20, alpha=0.7)
        
        axes[0].set_title('Latent Space (colored by Car ID)')
        axes[0].set_xlabel('Latent Dim 0')
        axes[0].set_ylabel('Latent Dim 1')
        axes[0].grid(True, alpha=0.3)
        
        # Color by slice index
        scatter = axes[1].scatter(embeddings[:, 0], embeddings[:, 1], 
                                c=slice_indices, s=20, alpha=0.7, cmap='viridis')
        axes[1].set_title('Latent Space (colored by Slice Index)')
        axes[1].set_xlabel('Latent Dim 0')
        axes[1].set_ylabel('Latent Dim 1')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Slice Index')
        
        plt.tight_layout()
        plt.show()
        
        return embeddings
    
    def interpolate_latent(self, 
                          slice1: torch.Tensor,
                          slice2: torch.Tensor,
                          num_steps: int = 8,
                          num_points: int = 500,
                          figsize: Tuple[int, int] = (20, 4)) -> None:
        """
        Interpolate between two slices in latent space.
        
        Args:
            slice1: First slice
            slice2: Second slice
            num_steps: Number of interpolation steps
            num_points: Number of points to generate
            figsize: Figure size
        """
        slice1 = slice1.unsqueeze(0).to(self.device) if slice1.dim() == 2 else slice1.to(self.device)
        slice2 = slice2.unsqueeze(0).to(self.device) if slice2.dim() == 2 else slice2.to(self.device)
        
        with torch.no_grad():
            # Encode both slices
            mu1, _ = self.model.encode(slice1)
            mu2, _ = self.model.encode(slice2)
            
            # Interpolate
            alphas = np.linspace(0, 1, num_steps)
            interpolated_slices = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                generated = self.model.decode(z_interp, num_points)
                interpolated_slices.append(generated[0].cpu().numpy())
        
        # Plot interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=figsize)
        
        for i, slice_points in enumerate(interpolated_slices):
            axes[i].scatter(slice_points[:, 0], slice_points[:, 1], s=1, alpha=0.7)
            axes[i].set_title(f'α = {alphas[i]:.2f}')
            axes[i].set_aspect('equal')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def compare_slices(slice1: np.ndarray, 
                  slice2: np.ndarray,
                  labels: Tuple[str, str] = ("Original", "Reconstruction"),
                  figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Standalone function to compare two slices.
    
    Args:
        slice1: First slice points
        slice2: Second slice points
        labels: Labels for the slices
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].scatter(slice1[:, 0], slice1[:, 1], s=2, alpha=0.7, c='blue')
    axes[0].set_title(labels[0])
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(slice2[:, 0], slice2[:, 1], s=2, alpha=0.7, c='red')
    axes[1].set_title(labels[1])
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_latent_space(embeddings: np.ndarray, 
                     labels: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot latent space embeddings.
    
    Args:
        embeddings: Latent embeddings, shape (N, latent_dim)
        labels: Optional labels for coloring
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        for i, label in enumerate(labels):
            ax.scatter(embeddings[i, 0], embeddings[i, 1], 
                      c=[label_to_color[label]], s=20, alpha=0.7, label=label)
        
        ax.legend()
    else:
        ax.scatter(embeddings[:, 0], embeddings[:, 1], s=20, alpha=0.7)
    
    ax.set_title('Latent Space Visualization')
    ax.set_xlabel('Latent Dim 0')
    ax.set_ylabel('Latent Dim 1')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """CLI interface for visualization tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PointFlow2D Visualization Tools")
    parser.add_argument("model_path", help="Path to trained model checkpoint")
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--num-samples", type=int, default=9, help="Number of samples to visualize")
    parser.add_argument("--num-points", type=int, default=500, help="Points per generated sample")
    
    args = parser.parse_args()
    
    print(f"🎨 Starting PointFlow2D visualization...")
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Data: {args.data_dir}")
    
    try:
        # Load model
        model = PointFlow2DVAE(latent_dim=args.latent_dim)
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Load dataset
        dataset = SliceDataset(args.data_dir, normalize=True)
        print(f"✅ Dataset loaded: {len(dataset)} slices")
        
        # Create visualizer
        visualizer = SliceVisualizer(model)
        
        # Generate samples
        print(f"\n🎲 Generating {args.num_samples} random samples...")
        samples = visualizer.generate_samples(args.num_samples, args.num_points)
        
        # Analyze latent space
        print(f"\n🧠 Analyzing latent space...")
        embeddings = visualizer.analyze_latent_space(dataset)
        
        # Show reconstruction for a random slice
        print(f"\n🔄 Showing reconstruction example...")
        idx = np.random.randint(len(dataset))
        item = dataset[idx]
        visualizer.visualize_reconstruction(item['points'])
        
        print(f"\n✅ Visualization completed!")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
