#!/usr/bin/env python3
"""
Quick test script to benchmark ODE performance on GPU vs CPU.
"""

import sys
import time
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_fixed import PointFlow2DVAE_Fixed

def benchmark_ode_performance():
    """Benchmark ODE performance GPU vs CPU."""
    
    print("🧪 ODE Performance Benchmark")
    print("=" * 50)
    
    # Test parameters
    batch_size = 4
    num_points = 300
    num_iterations = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create test data
    test_points = torch.randn(batch_size, num_points, 2, device=device)
    
    # Test configurations
    configs = [
        ("GPU ODE", False),
        ("CPU ODE", True)
    ]
    
    for config_name, force_cpu_ode in configs:
        print(f"\n🔬 Testing {config_name}...")
        
        # Create model
        model = PointFlow2DVAE_Fixed(
            latent_dim=64,
            encoder_hidden_dim=256,
            cnf_hidden_dim=128,
            latent_cnf_hidden_dim=128,
            use_latent_flow=True,
            force_cpu_ode=force_cpu_ode
        ).to(device)
        
        # Warmup
        print("  Warming up...")
        with torch.no_grad():
            _ = model.forward(test_points)
        
        # Benchmark
        times = []
        print(f"  Running {num_iterations} iterations...")
        
        for i in range(num_iterations):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                result = model.forward(test_points)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            iteration_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(iteration_time)
            print(f"    Iteration {i+1}: {iteration_time:.1f}ms")
        
        avg_time = sum(times) / len(times)
        print(f"  📊 Average: {avg_time:.1f}ms")
        
        # Quick memory check
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"  💾 Peak GPU memory: {memory_used:.1f}MB")
            torch.cuda.reset_peak_memory_stats()
    
    print("\n✅ Benchmark complete!")

def test_batch_sizes():
    """Test different batch sizes to find safe limits."""
    
    print("\n🔍 Batch Size Testing")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [4, 8, 16, 32]
    num_points = 300
    
    model = PointFlow2DVAE_Fixed(
        latent_dim=64,
        encoder_hidden_dim=256,
        cnf_hidden_dim=128,
        latent_cnf_hidden_dim=128,
        use_latent_flow=True,
        force_cpu_ode=False  # Test with GPU first
    ).to(device)
    
    for batch_size in batch_sizes:
        print(f"\n📦 Testing batch size: {batch_size}")
        
        try:
            test_points = torch.randn(batch_size, num_points, 2, device=device)
            
            start_time = time.time()
            with torch.no_grad():
                result = model.forward(test_points)
            end_time = time.time()
            
            iteration_time = (end_time - start_time) * 1000
            
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  ✅ Success: {iteration_time:.1f}ms, {memory_used:.1f}MB")
                torch.cuda.reset_peak_memory_stats()
            else:
                print(f"  ✅ Success: {iteration_time:.1f}ms")
                
        except RuntimeError as e:
            print(f"  ❌ Failed: {e}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    print("\n💡 Recommended batch sizes:")
    print("  - Conservative: 8")
    print("  - Aggressive: 16")
    print("  - RTX 4090 safe: 32")

if __name__ == "__main__":
    benchmark_ode_performance()
    test_batch_sizes()
