#!/usr/bin/env python3
"""
Inference script for 10-car trained model
Can be used after downloading model from RunPod
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import argparse
from scipy.spatial import cKDTree
sys.path.append('.')

# Import the EXACT SAME model
from src.models.pointflow2d_adapted import PointFlow2DAdaptedVAE
from src.training.dataset import SliceDataset
from torch.utils.data import DataLoader, Subset

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model with same config
    model = PointFlow2DAdaptedVAE(
        input_dim=2,
        latent_dim=config['latent_dim'],
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,
        solver='euler',
        solver_steps=10,
        force_cpu_ode=False
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device, config

def visualize_reconstructions(model, dataset, device, output_dir, num_samples=20):
    """Generate comprehensive reconstruction visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample evenly across dataset
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    # Collect all results
    all_chamfers = []
    all_coverages = []
    
    print(f"Generating {num_samples} reconstruction visualizations...")
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        points = item['points'].unsqueeze(0).to(device)
        num_points = item['num_points'].unsqueeze(0)
        car_id = item['car_id']
        slice_idx = item['slice_idx']
        
        # Create mask
        mask = torch.ones_like(points[..., 0])
        mask[0, num_points[0]:] = 0
        
        # Reconstruct
        with torch.no_grad():
            recon = model.reconstruct(points, mask, num_points)
        
        n_points = num_points[0].item()
        target_np = points[0, :n_points].cpu().numpy()
        recon_np = recon[0, :n_points].cpu().numpy()
        
        # Compute metrics
        chamfer = torch.cdist(points[0, :n_points], recon[0, :n_points]).min(1)[0].mean()
        chamfer += torch.cdist(recon[0, :n_points], points[0, :n_points]).min(1)[0].mean()
        chamfer = chamfer.item() / 2
        
        # Coverage
        threshold = 0.1
        dist_matrix = torch.cdist(points[0, :n_points], recon[0, :n_points])
        target_covered = (dist_matrix.min(dim=1)[0] < threshold).float().mean().item()
        pred_covered = (dist_matrix.min(dim=0)[0] < threshold).float().mean().item()
        
        all_chamfers.append(chamfer)
        all_coverages.append((target_covered, pred_covered))
        
        # Create individual visualization
        if i < 10:  # Save first 10
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original
            axes[0].scatter(target_np[:, 0], target_np[:, 1], alpha=0.6, s=2, c='blue')
            axes[0].set_title(f'Original - Car {car_id}, Slice {slice_idx}')
            axes[0].axis('equal')
            
            # Reconstructed
            axes[1].scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.6, s=2, c='red')
            axes[1].set_title(f'Reconstructed (Chamfer: {chamfer:.4f})')
            axes[1].axis('equal')
            
            # Overlay
            axes[2].scatter(target_np[:, 0], target_np[:, 1], alpha=0.5, s=2, c='blue', label='Original')
            axes[2].scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.5, s=2, c='red', label='Recon')
            axes[2].set_title(f'Coverage: T={target_covered:.1%} R={pred_covered:.1%}')
            axes[2].axis('equal')
            axes[2].legend()
            
            # Error heatmap
            tree = cKDTree(target_np)
            distances, _ = tree.query(recon_np)
            scatter = axes[3].scatter(recon_np[:, 0], recon_np[:, 1], 
                                    c=distances, cmap='hot', s=2, 
                                    vmin=0, vmax=0.1)
            axes[3].set_title('Point-wise Error')
            axes[3].axis('equal')
            plt.colorbar(scatter, ax=axes[3])
            
            plt.tight_layout()
            plt.savefig(output_dir / f'reconstruction_{i:03d}.png', dpi=150)
            plt.close()
    
    # Summary statistics
    chamfers = np.array(all_chamfers)
    target_covs = np.array([c[0] for c in all_coverages])
    pred_covs = np.array([c[1] for c in all_coverages])
    
    print(f"\n📊 Reconstruction Statistics ({num_samples} samples):")
    print(f"   Chamfer Distance:")
    print(f"      Mean: {chamfers.mean():.4f} ± {chamfers.std():.4f}")
    print(f"      Min: {chamfers.min():.4f}, Max: {chamfers.max():.4f}")
    print(f"   Target Coverage: {target_covs.mean():.1%} ± {target_covs.std():.1%}")
    print(f"   Recon Coverage: {pred_covs.mean():.1%} ± {pred_covs.std():.1%}")
    
    # Distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(chamfers, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(chamfers.mean(), color='red', linestyle='--', label=f'Mean: {chamfers.mean():.4f}')
    axes[0].set_xlabel('Chamfer Distance')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Chamfer Distance Distribution')
    axes[0].legend()
    
    axes[1].hist(target_covs, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Target Coverage')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Target Point Coverage')
    
    axes[2].hist(pred_covs, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[2].set_xlabel('Reconstruction Coverage')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Reconstruction Point Coverage')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistics.png', dpi=150)
    plt.close()
    
    # Save results
    results = {
        'num_samples': num_samples,
        'chamfer_mean': float(chamfers.mean()),
        'chamfer_std': float(chamfers.std()),
        'chamfer_min': float(chamfers.min()),
        'chamfer_max': float(chamfers.max()),
        'target_coverage_mean': float(target_covs.mean()),
        'pred_coverage_mean': float(pred_covs.mean()),
    }
    
    with open(output_dir / 'inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on trained 10-car model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint (best_model.pth)')
    parser.add_argument('--data-dir', type=str, default='data/training_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--num-cars', type=int, default=10,
                       help='Number of cars to use from dataset')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, device, config = load_model(args.checkpoint)
    print(f"Model loaded! Architecture: PointFlow2DAdaptedVAE")
    print(f"Config: {config}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = SliceDataset(
        data_directory=args.data_dir,
        normalize=True,
        max_points=1000,
        min_points=10
    )
    
    # Use subset for specified number of cars
    total_slices = len(dataset)
    slices_per_car = total_slices // dataset.loader.get_num_cars()
    num_slices = min(slices_per_car * args.num_cars, total_slices)
    dataset = Subset(dataset, list(range(num_slices)))
    
    print(f"Using {len(dataset)} slices from {args.num_cars} cars")
    
    # Run inference
    visualize_reconstructions(model, dataset, device, args.output_dir, args.num_samples)

if __name__ == "__main__":
    main()
