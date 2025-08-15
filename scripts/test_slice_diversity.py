#!/usr/bin/env python3
"""
Test model performance on different types of slices.
Important to ensure the model generalizes beyond single_slice_test.npy
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_cnf import PointFlow2DCNF

def analyze_slice(points):
    """Analyze slice characteristics."""
    if len(points) == 0:
        return None
    
    # Basic statistics
    num_points = len(points)
    centroid = points.mean(dim=0)
    
    # Bounding box
    min_coords = points.min(dim=0)[0]
    max_coords = points.max(dim=0)[0]
    bbox_size = max_coords - min_coords
    aspect_ratio = bbox_size[1] / (bbox_size[0] + 1e-6)
    
    # Spread
    distances = torch.norm(points - centroid, dim=1)
    mean_dist = distances.mean()
    std_dist = distances.std()
    
    # Density estimate (points per unit area)
    bbox_area = bbox_size[0] * bbox_size[1]
    density = num_points / (bbox_area + 1e-6)
    
    return {
        'num_points': num_points,
        'aspect_ratio': aspect_ratio.item(),
        'mean_dist': mean_dist.item(),
        'std_dist': std_dist.item(),
        'density': density.item(),
        'bbox_width': bbox_size[0].item(),
        'bbox_height': bbox_size[1].item()
    }

def test_slice_diversity(data_path: str, device: str = 'cuda'):
    """Test on diverse slices."""
    
    print("\n🧪 SLICE DIVERSITY TEST")
    print("=" * 50)
    
    # Find available slices
    data_dir = Path(data_path)
    slice_files = list(data_dir.glob("*.npy"))[:20]  # Test up to 20 slices
    
    if len(slice_files) < 2:
        print("Not enough slices for diversity test!")
        return
    
    print(f"Found {len(slice_files)} slices to test")
    
    # Analyze slice characteristics
    slice_stats = []
    slices_data = []
    
    for slice_file in slice_files:
        try:
            points = torch.from_numpy(np.load(slice_file)).float()
            if points.ndim == 1:
                points = points.reshape(-1, 2)
            
            stats = analyze_slice(points)
            if stats:
                stats['filename'] = slice_file.name
                slice_stats.append(stats)
                slices_data.append(points)
        except:
            continue
    
    # Group slices by characteristics
    small_slices = [i for i, s in enumerate(slice_stats) if s['num_points'] < 200]
    medium_slices = [i for i, s in enumerate(slice_stats) if 200 <= s['num_points'] < 600]
    large_slices = [i for i, s in enumerate(slice_stats) if s['num_points'] >= 600]
    
    print(f"\nSlice distribution:")
    print(f"  Small (<200 points): {len(small_slices)}")
    print(f"  Medium (200-600 points): {len(medium_slices)}")
    print(f"  Large (>600 points): {len(large_slices)}")
    
    # Test decoder on different slice types
    latent_dim = 32
    decoder = PointFlow2DCNF(
        point_dim=2,
        context_dim=latent_dim,
        hidden_dim=64,
        solver='euler',
        solver_steps=20
    ).to(device)
    
    # Test each category
    categories = [
        ('Small', small_slices[:3]),
        ('Medium', medium_slices[:3]),
        ('Large', large_slices[:3])
    ]
    
    results = {}
    
    for category_name, indices in categories:
        if not indices:
            continue
            
        print(f"\n📊 Testing {category_name} slices...")
        category_results = []
        
        for idx in indices:
            points = slices_data[idx].to(device)
            stats = slice_stats[idx]
            
            # Normalize
            center = points.mean(dim=0)
            scale = (points - center).abs().max() * 1.1
            points_norm = (points - center) / scale
            
            # Fixed latent
            fixed_z = torch.randn(1, latent_dim).to(device)
            
            # Quick training
            optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3)
            
            losses = []
            for epoch in range(100):
                optimizer.zero_grad()
                
                generated = decoder.sample(fixed_z, len(points)).squeeze(0)
                
                # Chamfer loss
                dist_g2t = torch.cdist(generated, points_norm).min(dim=1)[0].mean()
                dist_t2g = torch.cdist(points_norm, generated).min(dim=1)[0].mean()
                loss = dist_g2t + dist_t2g
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
                optimizer.step()
                
                losses.append(loss.item())
            
            final_loss = losses[-1]
            convergence_epoch = losses.index(min(losses))
            
            category_results.append({
                'filename': stats['filename'],
                'num_points': stats['num_points'],
                'final_loss': final_loss,
                'best_loss': min(losses),
                'convergence_epoch': convergence_epoch,
                'aspect_ratio': stats['aspect_ratio'],
                'density': stats['density']
            })
            
            print(f"  {stats['filename']}: {stats['num_points']} points, loss={final_loss:.4f}")
        
        results[category_name] = category_results
    
    # Visualize results
    output_dir = Path("outputs/slice_diversity")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss vs slice characteristics
    plt.figure(figsize=(15, 10))
    
    # Loss vs number of points
    plt.subplot(2, 3, 1)
    all_results = []
    for cat_results in results.values():
        all_results.extend(cat_results)
    
    if all_results:
        num_points = [r['num_points'] for r in all_results]
        losses = [r['best_loss'] for r in all_results]
        plt.scatter(num_points, losses, alpha=0.7, s=100)
        plt.xlabel('Number of Points')
        plt.ylabel('Best Loss')
        plt.title('Performance vs Slice Size')
        plt.grid(True)
        
        # Loss vs aspect ratio
        plt.subplot(2, 3, 2)
        aspects = [r['aspect_ratio'] for r in all_results]
        plt.scatter(aspects, losses, alpha=0.7, s=100)
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Best Loss')
        plt.title('Performance vs Shape')
        plt.grid(True)
        
        # Loss vs density
        plt.subplot(2, 3, 3)
        densities = [r['density'] for r in all_results]
        plt.scatter(densities, losses, alpha=0.7, s=100)
        plt.xlabel('Point Density')
        plt.ylabel('Best Loss')
        plt.title('Performance vs Density')
        plt.grid(True)
        
        # Category comparison
        plt.subplot(2, 3, 4)
        category_means = {}
        for cat_name, cat_results in results.items():
            if cat_results:
                category_means[cat_name] = np.mean([r['best_loss'] for r in cat_results])
        
        if category_means:
            plt.bar(category_means.keys(), category_means.values())
            plt.xlabel('Slice Category')
            plt.ylabel('Mean Best Loss')
            plt.title('Performance by Category')
        
        # Convergence speed
        plt.subplot(2, 3, 5)
        conv_epochs = [r['convergence_epoch'] for r in all_results]
        plt.scatter(num_points, conv_epochs, alpha=0.7, s=100)
        plt.xlabel('Number of Points')
        plt.ylabel('Convergence Epoch')
        plt.title('Convergence Speed')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "diversity_results.png", dpi=150)
    plt.close()
    
    # Summary
    print("\n📊 DIVERSITY TEST SUMMARY")
    print("=" * 50)
    
    if all_results:
        mean_loss = np.mean([r['best_loss'] for r in all_results])
        std_loss = np.std([r['best_loss'] for r in all_results])
        
        print(f"Overall performance: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"Best slice: {min(all_results, key=lambda x: x['best_loss'])['filename']}")
        print(f"Worst slice: {max(all_results, key=lambda x: x['best_loss'])['filename']}")
        
        # Check if performance correlates with size
        if len(all_results) > 2:
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(num_points, losses)
            print(f"\nCorrelation (size vs loss): {corr:.3f} (p={p_value:.3f})")
            
            if abs(corr) > 0.5:
                print("⚠️  Performance significantly affected by slice size!")
            else:
                print("✅ Performance consistent across slice sizes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    test_slice_diversity(args.data_path, args.device)
