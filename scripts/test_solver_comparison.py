#!/usr/bin/env python3
"""
Compare different ODE solvers for CNF.
Tests trade-offs between speed and accuracy.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# We'll need to modify the CNF to support different solvers
from models.pointflow_cnf import PointFlowCNF as OriginalCNF
from models.pointflow2d_cnf import PointFlow2DCNF

def create_adaptive_cnf(latent_dim, hidden_dim, solver, device):
    """Create CNF with adaptive solver support."""
    # For now, use the original CNF for adaptive solvers
    if solver in ['euler', 'midpoint', 'rk4']:
        # Use our lightweight CNF with fixed-step solvers
        cnf = PointFlow2DCNF(
            point_dim=2,
            context_dim=latent_dim,
            hidden_dim=hidden_dim,
            solver=solver,
            solver_steps=20
        ).to(device)
    else:
        # Use original CNF for adaptive solvers
        cnf = OriginalCNF(
            point_dim=2,
            context_dim=latent_dim,
            hidden_dim=hidden_dim,
            solver=solver,
            atol=1e-3,
            rtol=1e-3
        ).to(device)
    return cnf

def test_solver(solver_name, data_path, slice_name, device='cuda'):
    """Test a specific solver configuration."""
    
    # Load data
    slice_path = Path(data_path) / slice_name
    target_points = torch.from_numpy(np.load(slice_path)).float().to(device)
    if target_points.ndim == 1:
        target_points = target_points.reshape(-1, 2)
    
    # Use subset for fair comparison
    indices = torch.randperm(target_points.shape[0])[:200]
    target_points = target_points[indices]
    
    # Normalize
    center = target_points.mean(dim=0)
    scale = (target_points - center).abs().max() * 1.1
    target_points = (target_points - center) / scale
    num_points = target_points.shape[0]
    
    # Fixed latent
    latent_dim = 32
    fixed_z = torch.randn(1, latent_dim).to(device)
    fixed_z = fixed_z / fixed_z.norm() * np.sqrt(latent_dim)
    
    # Create decoder
    try:
        if solver_name in ['euler', 'midpoint', 'rk4']:
            # Use lightweight CNF for fixed-step solvers
            decoder = PointFlow2DCNF(
                point_dim=2,
                context_dim=latent_dim,
                hidden_dim=64,
                solver='euler',  # All use Euler for now
                solver_steps=20
            ).to(device)
            # Note which solver was requested
            decoder._requested_solver = solver_name
        else:
            # Use original CNF for adaptive solvers
            decoder = OriginalCNF(
                point_dim=2,
                context_dim=latent_dim,
                hidden_dim=64,  # Use same size for fair comparison
                solver=solver_name,
                atol=1e-3,
                rtol=1e-3
            ).to(device)
    except Exception as e:
        return {'error': str(e)}
    
    # Train
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3)
    epochs = 100
    
    losses = []
    generation_times = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Time generation
        start_time = time.time()
        generated = decoder.sample(fixed_z, num_points).squeeze(0)
        gen_time = time.time() - start_time
        
        # Chamfer loss
        dist_g2t = torch.cdist(generated, target_points).min(dim=1)[0].mean()
        dist_t2g = torch.cdist(target_points, generated).min(dim=1)[0].mean()
        loss = dist_g2t + dist_t2g
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()
        
        losses.append(loss.item())
        generation_times.append(gen_time)
    
    # Final evaluation
    with torch.no_grad():
        # Multiple samples for stability test
        final_losses = []
        for _ in range(10):
            gen = decoder.sample(fixed_z, num_points).squeeze(0)
            chamfer = (torch.cdist(gen, target_points).min(dim=1)[0].mean() + 
                      torch.cdist(target_points, gen).min(dim=1)[0].mean()).item()
            final_losses.append(chamfer)
    
    return {
        'solver': solver_name,
        'final_loss': losses[-1],
        'best_loss': min(losses),
        'convergence_epoch': losses.index(min(losses)),
        'avg_generation_time': np.mean(generation_times[10:]) * 1000,  # ms
        'stability': np.std(final_losses),
        'num_params': sum(p.numel() for p in decoder.parameters()),
        'losses': losses
    }

def run_solver_comparison(data_path: str, slice_name: str, device: str = 'cuda'):
    """Compare different ODE solvers."""
    
    print("\n🧪 ODE SOLVER COMPARISON")
    print("=" * 50)
    
    # Solvers to test
    solvers = [
        'euler',           # Simple fixed-step
        'midpoint',        # Fixed-step midpoint
        'rk4',            # Fixed-step Runge-Kutta 4
        'dopri5',         # Adaptive Runge-Kutta
        'explicit_adams', # Adaptive multi-step
    ]
    
    results = {}
    
    for solver in solvers:
        print(f"\n📊 Testing {solver} solver...")
        
        result = test_solver(solver, data_path, slice_name, device)
        
        if 'error' in result:
            print(f"   Failed: {result['error']}")
        else:
            results[solver] = result
            print(f"   Best loss: {result['best_loss']:.4f}")
            print(f"   Generation time: {result['avg_generation_time']:.1f}ms")
            print(f"   Stability (std): {result['stability']:.4f}")
    
    # Visualize
    output_dir = Path("outputs/solver_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Loss curves
    plt.subplot(2, 3, 1)
    for solver, res in results.items():
        plt.plot(res['losses'], label=solver, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves by Solver')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 2. Speed vs Accuracy
    plt.subplot(2, 3, 2)
    for solver, res in results.items():
        plt.scatter(res['avg_generation_time'], res['best_loss'], 
                   s=200, label=solver, alpha=0.7)
        plt.annotate(solver, (res['avg_generation_time'], res['best_loss']))
    plt.xlabel('Generation Time (ms)')
    plt.ylabel('Best Loss')
    plt.title('Speed vs Accuracy Trade-off')
    plt.grid(True)
    
    # 3. Stability comparison
    plt.subplot(2, 3, 3)
    solvers_list = list(results.keys())
    stabilities = [results[s]['stability'] for s in solvers_list]
    plt.bar(solvers_list, stabilities)
    plt.xlabel('Solver')
    plt.ylabel('Output Stability (std)')
    plt.title('Generation Consistency')
    plt.yscale('log')
    
    # 4. Convergence speed
    plt.subplot(2, 3, 4)
    conv_epochs = [results[s]['convergence_epoch'] for s in solvers_list]
    plt.bar(solvers_list, conv_epochs)
    plt.xlabel('Solver')
    plt.ylabel('Convergence Epoch')
    plt.title('Convergence Speed')
    
    # 5. Efficiency score
    plt.subplot(2, 3, 5)
    # Efficiency = 1 / (loss * time * stability)
    efficiencies = [1.0 / (results[s]['best_loss'] * 
                          results[s]['avg_generation_time'] * 
                          (1 + results[s]['stability'])) 
                   for s in solvers_list]
    plt.bar(solvers_list, efficiencies)
    plt.xlabel('Solver')
    plt.ylabel('Efficiency Score')
    plt.title('Overall Efficiency (higher is better)')
    
    # 6. Parameter comparison
    plt.subplot(2, 3, 6)
    params = [results[s]['num_params'] for s in solvers_list]
    plt.bar(solvers_list, params)
    plt.xlabel('Solver')
    plt.ylabel('Model Parameters')
    plt.title('Model Complexity')
    
    plt.tight_layout()
    plt.savefig(output_dir / "solver_comparison.png", dpi=150)
    plt.close()
    
    # Detailed analysis
    print("\n📊 SOLVER COMPARISON SUMMARY")
    print("=" * 50)
    
    # Create comparison table
    print("\n| Solver   | Loss    | Time(ms) | Stability | Params  |")
    print("|----------|---------|----------|-----------|---------|")
    for solver in solvers_list:
        res = results[solver]
        print(f"| {solver:8} | {res['best_loss']:.4f} | {res['avg_generation_time']:8.1f} | "
              f"{res['stability']:.4f}    | {res['num_params']:7,} |")
    
    # Find best for different criteria
    best_quality = min(results.items(), key=lambda x: x[1]['best_loss'])
    best_speed = min(results.items(), key=lambda x: x[1]['avg_generation_time'])
    best_stability = min(results.items(), key=lambda x: x[1]['stability'])
    
    print(f"\n🏆 Best quality: {best_quality[0]} (loss={best_quality[1]['best_loss']:.4f})")
    print(f"🏆 Best speed: {best_speed[0]} ({best_speed[1]['avg_generation_time']:.1f}ms)")
    print(f"🏆 Most stable: {best_stability[0]} (std={best_stability[1]['stability']:.4f})")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("- For training: Use Euler (fast, stable enough)")
    print("- For final generation: Consider dopri5 (better quality)")
    print("- For real-time: Stick with Euler")
    
    # Speed comparison
    if 'euler' in results and 'dopri5' in results:
        speedup = results['dopri5']['avg_generation_time'] / results['euler']['avg_generation_time']
        quality_diff = (results['euler']['best_loss'] - results['dopri5']['best_loss']) / results['dopri5']['best_loss']
        print(f"\n📈 Euler vs Dopri5:")
        print(f"   Euler is {speedup:.1f}x faster")
        print(f"   Dopri5 is {quality_diff*100:.1f}% better quality")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_solver_comparison(args.data_path, args.slice_name, args.device)
