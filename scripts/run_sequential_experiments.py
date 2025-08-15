#!/usr/bin/env python3
"""
Sequential Experiments Runner - Tests 3 different configurations
Option A: Longer training with higher LR
Option B: Different solver steps
Option C: Different LR scheduler
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import datetime
import json
import time
sys.path.append('.')

from src.models.pointflow2d_adapted import PointFlow2DAdaptedVAE
from overfit_single_slice_STABLE import (
    compute_chamfer_distance, 
    improved_chamfer_loss,
    load_single_slice,
    create_final_visualization
)

# Base configuration
BASE_CONFIG = {
    'latent_dim': 128,
    'batch_size': 8,
    'gradient_clip': 1.0,
    'weight_decay': 1e-4,
    'early_stop_patience': 100,
}

def run_experiment(experiment_name, config, output_dir):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*60}")
    print(f"🔬 EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load data
    print("\n📂 Loading single slice...")
    target_points, data_path = load_single_slice()
    
    # Center and scale
    center = target_points.mean(axis=0)
    scale = target_points.std()
    target_points = (target_points - center) / scale
    
    target_points = torch.from_numpy(target_points).float().to(device)
    num_points = target_points.shape[0]
    print(f"✓ Loaded slice with {num_points} points")
    
    # Create model
    print(f"\n🏗️  Building model for {experiment_name}...")
    print(f"  Configuration: {json.dumps(config, indent=2)}")
    
    model = PointFlow2DAdaptedVAE(
        input_dim=2,
        latent_dim=config['latent_dim'],
        encoder_hidden_dim=256,
        cnf_hidden_dim=256,
        solver=config.get('solver', 'euler'),
        solver_steps=config.get('solver_steps', 10),
        use_deterministic_encoder=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    if config.get('scheduler_type') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, T_mult=2, eta_min=config['min_lr']
        )
    elif config.get('scheduler_type') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=config['min_lr']
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr']
        )
    
    # Training
    start_time = time.time()
    print(f"\n🚀 Starting training (target Chamfer: {config['target_chamfer']})")
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Metrics tracking
    metrics_history = {
        'chamfer': [],
        'coverage_target': [],
        'coverage_pred': [],
        'learning_rate': [],
        'pf_loss': []
    }
    
    # Create batched target
    target_batch = target_points.unsqueeze(0).repeat(config['batch_size'], 1, 1)
    
    pbar = tqdm(range(config['epochs']), desc=experiment_name)
    for epoch in pbar:
        # Training step
        optimizer.zero_grad()
        
        B, N, D = target_batch.shape
        z = model.encode(target_batch)
        y, log_det = model.point_cnf(target_batch, z, reverse=False)
        
        log_py = -0.5 * (y ** 2).sum(dim=-1) - 0.5 * D * np.log(2 * np.pi)
        log_py = log_py.sum(dim=1, keepdim=True)
        log_px = log_py + log_det
        loss = -log_px.mean()
        
        # Output regularization
        output_reg = 0.01 * (y ** 2).mean()
        loss = loss + output_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
        optimizer.step()
        
        # Update scheduler
        if config.get('scheduler_type') == 'plateau':
            scheduler.step(loss)
        else:
            scheduler.step()
        
        pointflow_loss = loss.item()
        
        # Compute metrics every 10 epochs
        if epoch % 10 == 0:
            recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
            
            # Check for explosion
            recon_scale = recon.abs().max().item()
            if recon_scale > 10.0:
                print(f"\n⚠️ WARNING: Scale explosion at epoch {epoch}: {recon_scale:.2f}")
                break
            
            chamfer_dist = compute_chamfer_distance(target_points, recon)
            total_loss, chamfer, coverage_loss, target_cov, pred_cov = improved_chamfer_loss(
                recon, target_points
            )
            
            loss_val = chamfer_dist
            losses.append(loss_val)
            
            # Track metrics
            metrics_history['chamfer'].append(chamfer)
            metrics_history['coverage_target'].append(target_cov)
            metrics_history['coverage_pred'].append(pred_cov)
            metrics_history['learning_rate'].append(scheduler.get_last_lr()[0])
            metrics_history['pf_loss'].append(pointflow_loss)
            
            # Update best
            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)
            else:
                patience_counter += 1
        
        pbar.set_postfix({
            'chamfer': f'{losses[-1] if losses else 0:.4f}',
            'best': f'{best_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.1e}'
        })
        
        # Early stopping
        if losses and losses[-1] < config['target_chamfer']:
            print(f"\n🎯 Target reached at epoch {epoch}!")
            break
            
        if patience_counter >= config['early_stop_patience']:
            print(f"\n⏰ Early stopping at epoch {epoch}")
            break
        
        # Save visualization at intervals
        if epoch % 200 == 0:
            print(f"\n📊 Epoch {epoch} - Chamfer: {losses[-1] if losses else 0:.4f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n📊 EXPERIMENT RESULTS: {experiment_name}")
    print(f"  🎯 Best Chamfer: {best_loss:.4f} at epoch {best_epoch}")
    print(f"  🎯 Target: {config['target_chamfer']}")
    print(f"  ⏱️  Training time: {training_time/60:.1f} minutes")
    print(f"  ✅ Success: {'✓ PASSED' if best_loss < config['target_chamfer'] else '✗ FAILED'}")
    
    # Load best model for final analysis
    if (output_dir / 'best_model.pth').exists():
        checkpoint = torch.load(output_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_recon = model.reconstruct(target_points.unsqueeze(0)).squeeze(0)
        total_loss, chamfer, coverage_loss, target_cov, pred_cov = improved_chamfer_loss(
            final_recon, target_points
        )
        
        print(f"\n📊 BEST MODEL METRICS:")
        print(f"  📐 Chamfer: {chamfer:.4f}")
        print(f"  📈 Coverage: Target {target_cov*100:.1f}%, Pred {pred_cov*100:.1f}%")
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'best_loss': float(best_loss),
        'best_epoch': int(best_epoch),
        'final_epoch': epoch,
        'training_time_minutes': training_time/60,
        'metrics_history': metrics_history,
        'success': best_loss < config['target_chamfer']
    }
    
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create final visualization
    if len(losses) > 0:
        create_final_visualization(target_points, final_recon, losses, best_epoch, output_dir)
    
    return results

def main():
    """Run all three experiments sequentially"""
    
    # Create main output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = Path(f"outputs/sequential_experiments_{timestamp}")
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # EXPERIMENT A: Longer training with higher LR
    config_a = {
        **BASE_CONFIG,
        'epochs': 2000,
        'learning_rate': 1e-3,
        'min_lr': 1e-4,
        'target_chamfer': 0.05,
        'scheduler_type': 'cosine'
    }
    
    output_dir_a = main_output_dir / 'experiment_a_longer_training'
    output_dir_a.mkdir(exist_ok=True)
    
    results_a = run_experiment("A: Longer Training + Higher LR", config_a, output_dir_a)
    all_results['experiment_a'] = results_a
    
    # EXPERIMENT B: Different solver steps
    solver_steps_configs = [
        {'steps': 5, 'name': 'B1: 5 Steps (Fast)'},
        {'steps': 10, 'name': 'B2: 10 Steps (Default)'},
        {'steps': 20, 'name': 'B3: 20 Steps (Accurate)'}
    ]
    
    all_results['experiment_b'] = {}
    
    for solver_config in solver_steps_configs:
        config_b = {
            **BASE_CONFIG,
            'epochs': 1000,
            'learning_rate': 5e-4,
            'min_lr': 1e-4,
            'target_chamfer': 0.05,
            'solver_steps': solver_config['steps'],
            'scheduler_type': 'cosine'
        }
        
        output_dir_b = main_output_dir / f"experiment_b_solver_{solver_config['steps']}_steps"
        output_dir_b.mkdir(exist_ok=True)
        
        results_b = run_experiment(solver_config['name'], config_b, output_dir_b)
        all_results['experiment_b'][f"steps_{solver_config['steps']}"] = results_b
    
    # EXPERIMENT C: Different LR scheduler
    config_c = {
        **BASE_CONFIG,
        'epochs': 1500,
        'learning_rate': 8e-4,
        'min_lr': 5e-5,
        'target_chamfer': 0.05,
        'scheduler_type': 'plateau'
    }
    
    output_dir_c = main_output_dir / 'experiment_c_plateau_scheduler'
    output_dir_c.mkdir(exist_ok=True)
    
    results_c = run_experiment("C: ReduceLROnPlateau Scheduler", config_c, output_dir_c)
    all_results['experiment_c'] = results_c
    
    # Save combined results
    with open(main_output_dir / 'all_experiments_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ALL EXPERIMENTS SUMMARY")
    print("="*60)
    
    print("\n🔬 Experiment A (Longer Training):")
    print(f"  Best Chamfer: {results_a['best_loss']:.4f}")
    print(f"  Success: {results_a['success']}")
    
    print("\n🔬 Experiment B (Solver Steps):")
    for steps, result in all_results['experiment_b'].items():
        print(f"  {steps}: Chamfer={result['best_loss']:.4f}, Success={result['success']}")
    
    print("\n🔬 Experiment C (Plateau Scheduler):")
    print(f"  Best Chamfer: {results_c['best_loss']:.4f}")
    print(f"  Success: {results_c['success']}")
    
    print(f"\n✅ All experiments complete!")
    print(f"📁 Results saved to: {main_output_dir}")
    
    # Create comparison plot
    create_comparison_plot(all_results, main_output_dir)

def create_comparison_plot(all_results, output_dir):
    """Create a comparison plot of all experiments"""
    plt.figure(figsize=(15, 5))
    
    # Extract data
    experiments = []
    chamfer_scores = []
    colors = []
    
    # Experiment A
    experiments.append('A: Longer\nTraining')
    chamfer_scores.append(all_results['experiment_a']['best_loss'])
    colors.append('blue')
    
    # Experiment B
    for steps in [5, 10, 20]:
        experiments.append(f'B: {steps}\nSteps')
        chamfer_scores.append(all_results['experiment_b'][f'steps_{steps}']['best_loss'])
        colors.append('green')
    
    # Experiment C
    experiments.append('C: Plateau\nScheduler')
    chamfer_scores.append(all_results['experiment_c']['best_loss'])
    colors.append('red')
    
    # Bar plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(experiments, chamfer_scores, color=colors, alpha=0.7)
    plt.axhline(y=0.05, color='black', linestyle='--', label='Target')
    plt.ylabel('Best Chamfer Distance')
    plt.title('Experiment Comparison')
    plt.xticks(rotation=0)
    plt.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars, chamfer_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.4f}', ha='center', va='bottom')
    
    # Training time comparison
    plt.subplot(1, 2, 2)
    times = []
    times.append(all_results['experiment_a']['training_time_minutes'])
    for steps in [5, 10, 20]:
        times.append(all_results['experiment_b'][f'steps_{steps}']['training_time_minutes'])
    times.append(all_results['experiment_c']['training_time_minutes'])
    
    plt.bar(experiments, times, color=colors, alpha=0.7)
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Efficiency')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiments_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
