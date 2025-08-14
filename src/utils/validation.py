"""
Validation utilities for PointFlow2D training.
Implements meaningful validation strategies from original PointFlow adapted for 2D.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Comprehensive training monitor implementing PointFlow's validation strategies.
    """
    
    def __init__(self, 
                 save_dir: str,
                 model_name: str = "pointflow2d",
                 save_frequency: int = 10,
                 visualization_frequency: int = 5):
        """
        Initialize training monitor.
        
        Args:
            save_dir: Directory to save monitoring outputs
            model_name: Name of the model for file naming
            save_frequency: How often to save validation results (epochs)
            visualization_frequency: How often to save visualizations (epochs)
        """
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.save_frequency = save_frequency
        self.viz_frequency = visualization_frequency
        
        # Create subdirectories
        self.outputs_dir = self.save_dir / "validation_outputs"
        self.plots_dir = self.save_dir / "plots"
        self.slices_dir = self.save_dir / "decoded_slices"
        
        for dir_path in [self.outputs_dir, self.plots_dir, self.slices_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'recon_loss': [],
            'prior_loss': [],
            'kl_loss': [],
            'latent_magnitude': [],
            'gradient_norm': [],
            'learning_rate': []
        }
        
        # Validation metrics
        self.validation_history = {
            'epoch': [],
            'reconstruction_quality': [],
            'sampling_quality': [],
            'latent_space_health': []
        }
        
        logger.info(f"TrainingMonitor initialized: {self.save_dir}")
    
    def log_training_step(self, 
                         epoch: int,
                         loss_dict: Dict[str, float],
                         model: nn.Module,
                         optimizer: torch.optim.Optimizer):
        """
        Log training metrics for one epoch.
        
        Args:
            epoch: Current epoch number
            loss_dict: Dictionary containing all loss components
            model: The model being trained
            optimizer: Optimizer being used
        """
        # Record basic metrics
        self.history['epoch'].append(epoch)
        self.history['total_loss'].append(loss_dict.get('total_loss', 0.0))
        self.history['recon_loss'].append(loss_dict.get('recon_loss', 0.0))
        self.history['prior_loss'].append(loss_dict.get('prior_loss', 0.0))
        self.history['kl_loss'].append(loss_dict.get('kl_loss', 0.0))
        
        # Compute additional diagnostics
        grad_norm = self._compute_gradient_norm(model)
        self.history['gradient_norm'].append(grad_norm)
        
        # Get learning rate
        lr = optimizer.param_groups[0]['lr']
        self.history['learning_rate'].append(lr)
        
        # Compute latent magnitude (if encoder exists)
        if hasattr(model, 'encoder'):
            with torch.no_grad():
                # Use a small batch for diagnostic
                dummy_input = torch.randn(4, 50, 2).to(next(model.parameters()).device)
                mu, logvar = model.encode(dummy_input)
                latent_mag = torch.norm(mu, dim=1).mean().item()
                self.history['latent_magnitude'].append(latent_mag)
        else:
            self.history['latent_magnitude'].append(0.0)
    
    def validate_reconstruction(self, 
                              model: nn.Module,
                              validation_slice: torch.Tensor,
                              epoch: int) -> Dict[str, float]:
        """
        Validate reconstruction quality on a specific slice.
        
        Args:
            model: Model to validate
            validation_slice: Single slice tensor (1, num_points, 2)
            epoch: Current epoch
            
        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        
        with torch.no_grad():
            # Get device
            device = next(model.parameters()).device
            validation_slice = validation_slice.to(device)
            
            # Forward pass
            result = model.forward(validation_slice)
            
            # Reconstruction
            z = result['z']
            num_points = validation_slice.shape[1]
            reconstructed = model.decode(z, num_points)
            
            # Compute metrics
            mse_loss = torch.nn.functional.mse_loss(reconstructed, validation_slice)
            chamfer_dist = self._compute_chamfer_distance(reconstructed, validation_slice)
            
            metrics = {
                'reconstruction_mse': mse_loss.item(),
                'chamfer_distance': chamfer_dist.item(),
                'latent_magnitude': torch.norm(z).item(),
                'recon_loss': result['log_likelihood'].mean().item(),
                'prior_loss': result['log_prior'].mean().item(),
                'kl_loss': result['kl_loss'].mean().item()
            }
            
            # Save visualization if it's time
            if epoch % self.viz_frequency == 0:
                self._save_reconstruction_visualization(
                    validation_slice, reconstructed, epoch, metrics
                )
            
            # Save decoded slice for RunPod fetching
            if epoch % self.save_frequency == 0:
                self._save_decoded_slice(reconstructed, epoch)
        
        model.train()
        return metrics
    
    def validate_sampling(self, 
                         model: nn.Module,
                         num_samples: int = 5,
                         num_points: int = 100,
                         epoch: int = 0) -> Dict[str, float]:
        """
        Validate sampling quality.
        
        Args:
            model: Model to validate
            num_samples: Number of samples to generate
            num_points: Number of points per sample
            epoch: Current epoch
            
        Returns:
            Dictionary of sampling metrics
        """
        model.eval()
        
        with torch.no_grad():
            device = next(model.parameters()).device
            
            # Generate samples
            samples = model.sample(num_samples, num_points, device)
            
            # Compute metrics
            point_variance = torch.var(samples, dim=1).mean().item()
            spatial_coverage = self._compute_spatial_coverage(samples)
            
            metrics = {
                'point_variance': point_variance,
                'spatial_coverage': spatial_coverage,
                'sample_diversity': self._compute_sample_diversity(samples)
            }
            
            # Save sample visualization
            if epoch % self.viz_frequency == 0:
                self._save_sampling_visualization(samples, epoch, metrics)
        
        model.train()
        return metrics
    
    def detect_training_issues(self, recent_history: int = 10) -> List[str]:
        """
        Detect potential training issues based on recent history.
        
        Args:
            recent_history: Number of recent epochs to analyze
            
        Returns:
            List of detected issues
        """
        issues = []
        
        if len(self.history['total_loss']) < recent_history:
            return issues
        
        recent_losses = self.history['total_loss'][-recent_history:]
        recent_grads = self.history['gradient_norm'][-recent_history:]
        recent_latent = self.history['latent_magnitude'][-recent_history:]
        
        # Check for loss explosion
        if any(loss > 1e6 for loss in recent_losses):
            issues.append("CRITICAL: Loss explosion detected")
        
        # Check for gradient explosion
        if any(grad > 100 for grad in recent_grads):
            issues.append("WARNING: Gradient explosion detected")
        
        # Check for gradient vanishing
        if all(grad < 1e-6 for grad in recent_grads):
            issues.append("WARNING: Gradient vanishing detected")
        
        # Check for latent collapse
        if all(mag < 0.1 for mag in recent_latent):
            issues.append("WARNING: Latent space collapse detected")
        
        # Check for latent explosion
        if any(mag > 50 for mag in recent_latent):
            issues.append("WARNING: Latent space explosion detected")
        
        # Check for loss stagnation
        if len(set(f"{loss:.4f}" for loss in recent_losses)) == 1:
            issues.append("WARNING: Loss stagnation detected")
        
        return issues
    
    def save_training_plots(self, epoch: int):
        """
        Save comprehensive training plots.
        
        Args:
            epoch: Current epoch number
        """
        if len(self.history['epoch']) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16)
        
        # Loss components
        ax = axes[0, 0]
        epochs = self.history['epoch']
        ax.plot(epochs, self.history['total_loss'], label='Total Loss', linewidth=2)
        ax.plot(epochs, self.history['recon_loss'], label='Reconstruction', alpha=0.7)
        ax.plot(epochs, self.history['prior_loss'], label='Prior', alpha=0.7)
        ax.plot(epochs, self.history['kl_loss'], label='KL', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Gradient norm
        ax = axes[0, 1]
        ax.plot(epochs, self.history['gradient_norm'], color='green', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)
        
        # Latent magnitude
        ax = axes[0, 2]
        ax.plot(epochs, self.history['latent_magnitude'], color='purple', linewidth=2)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Target (1.0)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Latent Magnitude')
        ax.set_title('Latent Space Health')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 0]
        ax.plot(epochs, self.history['learning_rate'], color='orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        # Loss ratios
        ax = axes[1, 1]
        if len(self.history['recon_loss']) > 0 and all(r > 0 for r in self.history['recon_loss']):
            kl_ratios = [k/r for k, r in zip(self.history['kl_loss'], self.history['recon_loss'])]
            ax.plot(epochs, kl_ratios, color='red', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('KL/Reconstruction Ratio')
            ax.set_title('Loss Balance')
            ax.grid(True, alpha=0.3)
        
        # Recent loss trend
        ax = axes[1, 2]
        recent_epochs = epochs[-20:] if len(epochs) > 20 else epochs
        recent_losses = self.history['total_loss'][-20:] if len(self.history['total_loss']) > 20 else self.history['total_loss']
        ax.plot(recent_epochs, recent_losses, color='blue', linewidth=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Recent Loss Trend')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"training_progress_epoch_{epoch:04d}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training plots: {plot_path}")
    
    def save_validation_summary(self, epoch: int):
        """
        Save validation summary for RunPod analysis.
        
        Args:
            epoch: Current epoch number
        """
        summary = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'training_history': {
                'epochs': len(self.history['epoch']),
                'latest_losses': {
                    'total': self.history['total_loss'][-1] if self.history['total_loss'] else None,
                    'reconstruction': self.history['recon_loss'][-1] if self.history['recon_loss'] else None,
                    'prior': self.history['prior_loss'][-1] if self.history['prior_loss'] else None,
                    'kl': self.history['kl_loss'][-1] if self.history['kl_loss'] else None,
                },
                'gradient_norm': self.history['gradient_norm'][-1] if self.history['gradient_norm'] else None,
                'latent_magnitude': self.history['latent_magnitude'][-1] if self.history['latent_magnitude'] else None,
            },
            'detected_issues': self.detect_training_issues(),
            'validation_metrics': self.validation_history
        }
        
        summary_path = self.outputs_dir / f"validation_summary_epoch_{epoch:04d}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved validation summary: {summary_path}")
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute gradient norm for monitoring."""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def _compute_chamfer_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute simplified Chamfer distance for 2D points."""
        # pred: (batch, num_points, 2)
        # target: (batch, num_points, 2)
        
        # Compute pairwise distances
        pred_expanded = pred.unsqueeze(2)  # (batch, num_points, 1, 2)
        target_expanded = target.unsqueeze(1)  # (batch, 1, num_points, 2)
        
        dists = torch.norm(pred_expanded - target_expanded, dim=3)  # (batch, num_points, num_points)
        
        # Forward Chamfer: min distance from pred to target
        forward_chamfer = torch.min(dists, dim=2)[0].mean()
        
        # Backward Chamfer: min distance from target to pred
        backward_chamfer = torch.min(dists, dim=1)[0].mean()
        
        return (forward_chamfer + backward_chamfer) / 2
    
    def _compute_spatial_coverage(self, samples: torch.Tensor) -> float:
        """Compute spatial coverage of generated samples."""
        # Flatten samples to (total_points, 2)
        points = samples.view(-1, 2)
        
        # Compute bounding box area
        min_coords = torch.min(points, dim=0)[0]
        max_coords = torch.max(points, dim=0)[0]
        coverage = torch.prod(max_coords - min_coords).item()
        
        return coverage
    
    def _compute_sample_diversity(self, samples: torch.Tensor) -> float:
        """Compute diversity of samples using pairwise distances."""
        batch_size = samples.shape[0]
        if batch_size < 2:
            return 0.0
        
        # Compute pairwise distances between samples
        total_diversity = 0.0
        count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # Simple diversity metric: mean pairwise distance
                diversity = torch.norm(samples[i] - samples[j], dim=1).mean().item()
                total_diversity += diversity
                count += 1
        
        return total_diversity / count if count > 0 else 0.0
    
    def _save_reconstruction_visualization(self, 
                                         original: torch.Tensor,
                                         reconstructed: torch.Tensor,
                                         epoch: int,
                                         metrics: Dict[str, float]):
        """Save reconstruction comparison visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original
        ax = axes[0]
        original_np = original[0].cpu().numpy()
        ax.scatter(original_np[:, 0], original_np[:, 1], alpha=0.6, s=10)
        ax.set_title('Original Slice')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Reconstructed
        ax = axes[1]
        recon_np = reconstructed[0].cpu().numpy()
        ax.scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.6, s=10, color='red')
        ax.set_title(f'Reconstructed (Epoch {epoch})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add metrics as text
        metrics_text = f"MSE: {metrics['reconstruction_mse']:.4f}\n"
        metrics_text += f"Chamfer: {metrics['chamfer_distance']:.4f}\n"
        metrics_text += f"Latent Mag: {metrics['latent_magnitude']:.4f}"
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        save_path = self.plots_dir / f"reconstruction_epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_sampling_visualization(self, 
                                   samples: torch.Tensor,
                                   epoch: int,
                                   metrics: Dict[str, float]):
        """Save sampling visualization."""
        fig, axes = plt.subplots(1, min(5, samples.shape[0]), figsize=(15, 3))
        if samples.shape[0] == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i < samples.shape[0]:
                sample_np = samples[i].cpu().numpy()
                ax.scatter(sample_np[:, 0], sample_np[:, 1], alpha=0.6, s=10)
                ax.set_title(f'Sample {i+1}')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
        
        # Add metrics
        metrics_text = f"Point Var: {metrics['point_variance']:.4f}\n"
        metrics_text += f"Coverage: {metrics['spatial_coverage']:.4f}\n" 
        metrics_text += f"Diversity: {metrics['sample_diversity']:.4f}"
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        save_path = self.plots_dir / f"sampling_epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_decoded_slice(self, decoded_slice: torch.Tensor, epoch: int):
        """Save decoded slice for RunPod fetching."""
        slice_data = {
            'epoch': epoch,
            'points': decoded_slice[0].cpu().numpy().tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = self.slices_dir / f"decoded_slice_epoch_{epoch:04d}.json"
        with open(save_path, 'w') as f:
            json.dump(slice_data, f)
        
        # Also save as numpy for easy loading
        np_save_path = self.slices_dir / f"decoded_slice_epoch_{epoch:04d}.npy"
        np.save(np_save_path, decoded_slice[0].cpu().numpy())


def setup_validation_slice(data_dir: str, 
                          slice_filename: str = None) -> torch.Tensor:
    """
    Setup a validation slice for consistent monitoring.
    
    Args:
        data_dir: Directory containing slice data
        slice_filename: Specific slice file to use, or None for random
        
    Returns:
        Validation slice tensor
    """
    data_path = Path(data_dir)
    
    if slice_filename:
        slice_path = data_path / slice_filename
        if not slice_path.exists():
            raise FileNotFoundError(f"Slice file not found: {slice_path}")
    else:
        # Get a random slice file
        slice_files = list(data_path.glob("*.npy"))
        if not slice_files:
            raise FileNotFoundError(f"No .npy files found in {data_path}")
        slice_path = slice_files[0]  # Use first file for consistency
    
    # Load slice data
    slice_data = np.load(slice_path, allow_pickle=True)
    
    # Handle the case where file contains multiple slices (array of slices)
    if slice_data.dtype == object and len(slice_data.shape) == 1:
        # File contains multiple slices, select the first one
        actual_slice = slice_data[0]
        logger.info(f"Loaded file with {len(slice_data)} slices, using slice 0")
    else:
        # File contains a single slice
        actual_slice = slice_data
    
    # Normalize to [-1, 1] range
    min_coords = actual_slice.min(axis=0)
    max_coords = actual_slice.max(axis=0)
    normalized_slice = 2 * (actual_slice - min_coords) / (max_coords - min_coords) - 1
    
    # Convert to tensor and add batch dimension
    slice_tensor = torch.FloatTensor(normalized_slice).unsqueeze(0)
    
    logger.info(f"Setup validation slice: {slice_path.name} (slice shape: {actual_slice.shape})")
    return slice_tensor
