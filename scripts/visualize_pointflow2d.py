#!/usr/bin/env python3
"""
Visualization and analysis script for trained PointFlow2D models.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.visualization import SliceVisualizer
from models.pointflow2d import PointFlow2DVAE
from training.dataset import SliceDataset
import torch


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(description="Visualize PointFlow2D results")
    parser.add_argument("model_path", help="Path to trained model checkpoint")
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--num-samples", type=int, default=9, help="Number of samples to generate")
    parser.add_argument("--num-points", type=int, default=500, help="Points per generated sample")
    parser.add_argument("--interpolation", action="store_true", help="Show latent interpolation")
    parser.add_argument("--latent-analysis", action="store_true", help="Analyze latent space")
    parser.add_argument("--reconstruction", action="store_true", help="Show reconstruction examples")
    parser.add_argument("--all", action="store_true", help="Run all visualizations")
    
    args = parser.parse_args()
    
    print("🎨 PointFlow2D Visualization & Analysis")
    print("=" * 50)
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Data: {args.data_dir}")
    print(f"🧠 Latent dimension: {args.latent_dim}")
    print("=" * 50)
    
    try:
        # Load model
        print(f"\n📦 Loading model...")
        model = PointFlow2DVAE(latent_dim=args.latent_dim)
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ Model loaded from epoch {epoch}")
        
        # Load dataset
        print(f"\n📂 Loading dataset...")
        dataset = SliceDataset(args.data_dir, normalize=True, min_points=50)
        stats = dataset.get_statistics()
        print(f"✅ Dataset loaded: {stats['total_slices']} slices from {stats['unique_cars']} cars")
        
        # Create visualizer
        visualizer = SliceVisualizer(model)
        
        # Run visualizations based on flags
        if args.all or args.num_samples > 0:
            print(f"\n🎲 Generating {args.num_samples} random samples...")
            samples = visualizer.generate_samples(args.num_samples, args.num_points)
            print(f"✅ Generated samples with shape: {[s.shape for s in samples[:3]]}...")
        
        if args.all or args.latent_analysis:
            print(f"\n🧠 Analyzing latent space...")
            embeddings = visualizer.analyze_latent_space(dataset, num_samples=min(200, len(dataset)))
            print(f"✅ Analyzed {len(embeddings)} embeddings")
        
        if args.all or args.reconstruction:
            print(f"\n🔄 Showing reconstruction examples...")
            # Show multiple reconstruction examples
            for i in range(3):
                idx = np.random.randint(len(dataset))
                item = dataset[idx]
                print(f"   Example {i+1}: Car {item['car_id']}, Slice {item['slice_idx']}")
                visualizer.visualize_reconstruction(item['points'], num_samples=3)
        
        if args.all or args.interpolation:
            print(f"\n🔀 Showing latent space interpolation...")
            # Get two random slices
            idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
            slice1 = dataset[idx1]['points']
            slice2 = dataset[idx2]['points']
            
            print(f"   Interpolating between:")
            print(f"     Slice 1: Car {dataset[idx1]['car_id']}, Slice {dataset[idx1]['slice_idx']}")
            print(f"     Slice 2: Car {dataset[idx2]['car_id']}, Slice {dataset[idx2]['slice_idx']}")
            
            visualizer.interpolate_latent(slice1, slice2, num_steps=8, num_points=args.num_points)
        
        # Summary statistics
        print(f"\n📊 Model Performance Summary:")
        if 'train_loss' in checkpoint:
            losses = checkpoint.get('train_losses', [])
            if losses:
                print(f"   Final training loss: {losses[-1]:.4f}")
        
        # Quick quality check
        print(f"\n🔍 Quality Check:")
        idx = np.random.randint(len(dataset))
        item = dataset[idx]
        points = item['points'].unsqueeze(0)
        
        with torch.no_grad():
            result = visualizer.model.forward(points)
            kl_loss = result['kl_loss'].item()
            recon = result['x_recon'][0]
            
        print(f"   Sample KL loss: {kl_loss:.4f}")
        print(f"   Original points: {points.shape[1]}")
        print(f"   Reconstructed points: {recon.shape[0]}")
        print(f"   Point range: [{recon.min():.3f}, {recon.max():.3f}]")
        
        print(f"\n✅ Visualization completed successfully!")
        print(f"\n💡 Tips:")
        print(f"   - Use --all flag to run all visualizations")
        print(f"   - Adjust --num-points to change generation resolution")
        print(f"   - Check training curves in the model directory")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
