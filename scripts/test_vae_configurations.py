#!/usr/bin/env python3
"""
Comprehensive VAE Configuration Test
Tests different combinations of encoder, decoder, and latent CNF configurations.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.encoder import PointNet2DEncoder, reparameterize
from models.pointflow2d_cnf import PointFlow2DCNF
from models.latent_cnf import LatentCNF

class SimpleVAE(nn.Module):
    """Simple VAE for testing different configurations."""
    
    def __init__(self, encoder_hidden, decoder_hidden, latent_dim, use_latent_cnf=False):
        super().__init__()
        
        self.encoder = PointNet2DEncoder(
            input_dim=2,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden
        )
        
        self.decoder = PointFlow2DCNF(
            point_dim=2,
            context_dim=latent_dim,
            hidden_dim=decoder_hidden,
            solver='euler',
            solver_steps=20
        )
        
        self.use_latent_cnf = use_latent_cnf
        if use_latent_cnf:
            self.latent_cnf = LatentCNF(
                latent_dim=latent_dim,
                hidden_dim=32,  # Smaller for faster testing
                solver='euler',  # Use euler for speed
                atol=1e-2,      # Relaxed tolerances
                rtol=1e-2
            )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z, num_points):
        return self.decoder.sample(z, num_points).squeeze(0)
    
    def forward(self, x, beta=1.0):
        batch_size, num_points = x.shape[:2]
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = reparameterize(mu, logvar)
        
        # Decode
        recon = self.decoder.sample(z, num_points).squeeze(0)
        
        # Losses
        # Reconstruction loss
        dist_g2t = torch.cdist(recon, x.squeeze(0)).min(dim=1)[0].mean()
        dist_t2g = torch.cdist(x.squeeze(0), recon).min(dim=1)[0].mean()
        recon_loss = dist_g2t + dist_t2g
        
        # KL divergence
        if self.use_latent_cnf:
            # Transform z through latent CNF
            w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
            # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) + delta_log_pw
            kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) - delta_log_pw.sum()
        else:
            # Standard VAE KL
            kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        
        kl_loss = kl_loss / batch_size
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': mu,
            'logvar': logvar
        }

def test_configuration(config, data_path, slice_name, device='cuda'):
    """Test a specific VAE configuration."""
    
    # Load data
    slice_path = Path(data_path) / slice_name
    target_points = torch.from_numpy(np.load(slice_path)).float().to(device)
    if target_points.ndim == 1:
        target_points = target_points.reshape(-1, 2)
    
    # Normalize
    center = target_points.mean(dim=0)
    scale = (target_points - center).abs().max() * 1.1
    target_points = (target_points - center) / scale
    num_points = target_points.shape[0]
    
    # Create VAE
    vae = SimpleVAE(
        encoder_hidden=config['encoder_hidden'],
        decoder_hidden=config['decoder_hidden'],
        latent_dim=config['latent_dim'],
        use_latent_cnf=config['use_latent_cnf']
    ).to(device)
    
    total_params = sum(p.numel() for p in vae.parameters())
    
    # Train with beta scheduling
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    epochs = 200
    
    # Beta schedule (warm-up)
    beta_schedule = np.concatenate([
        np.linspace(0.0, 1.0, 100),  # Warm-up
        np.ones(100)                  # Full beta
    ])
    
    losses = []
    recon_losses = []
    kl_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        beta = beta_schedule[epoch]
        out = vae(target_points.unsqueeze(0), beta=beta)
        
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
        optimizer.step()
        
        losses.append(out['loss'].item())
        recon_losses.append(out['recon_loss'].item())
        kl_losses.append(out['kl_loss'].item())
    
    # Evaluate final quality
    with torch.no_grad():
        # Reconstruction quality
        mu, _ = vae.encode(target_points.unsqueeze(0))
        recon = vae.decode(mu, num_points)
        final_recon_loss = (torch.cdist(recon, target_points).min(dim=1)[0].mean() + 
                           torch.cdist(target_points, recon).min(dim=1)[0].mean()).item()
        
        # Generation quality (from prior)
        z_prior = torch.randn(1, config['latent_dim']).to(device)
        gen_from_prior = vae.decode(z_prior, num_points)
        
        # Latent space metrics
        latent_mean = mu.mean().item()
        latent_std = mu.std().item()
    
    return {
        'config': config,
        'total_params': total_params,
        'final_loss': losses[-1],
        'final_recon_loss': final_recon_loss,
        'final_kl_loss': kl_losses[-1],
        'best_recon_loss': min(recon_losses),
        'latent_mean': latent_mean,
        'latent_std': latent_std,
        'losses': losses,
        'recon_losses': recon_losses,
        'kl_losses': kl_losses
    }

def run_vae_configuration_test(data_path: str, slice_name: str, device: str = 'cuda'):
    """Test comprehensive VAE configurations."""
    
    print("\n🧪 VAE CONFIGURATION TEST")
    print("=" * 50)
    
    # Configurations to test
    configs = [
        # Baseline
        {'encoder_hidden': 128, 'decoder_hidden': 64, 'latent_dim': 32, 'use_latent_cnf': False},
        
        # Smaller
        {'encoder_hidden': 64, 'decoder_hidden': 32, 'latent_dim': 16, 'use_latent_cnf': False},
        
        # Larger
        {'encoder_hidden': 256, 'decoder_hidden': 128, 'latent_dim': 64, 'use_latent_cnf': False},
        
        # With Latent CNF
        {'encoder_hidden': 128, 'decoder_hidden': 64, 'latent_dim': 32, 'use_latent_cnf': True},
        
        # Bottleneck
        {'encoder_hidden': 128, 'decoder_hidden': 64, 'latent_dim': 8, 'use_latent_cnf': False},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n📊 Testing configuration {i+1}/{len(configs)}:")
        print(f"   Encoder: {config['encoder_hidden']}, Decoder: {config['decoder_hidden']}")
        print(f"   Latent: {config['latent_dim']}, Latent CNF: {config['use_latent_cnf']}")
        
        try:
            result = test_configuration(config, data_path, slice_name, device)
            results.append(result)
            print(f"   Parameters: {result['total_params']:,}")
            print(f"   Final recon loss: {result['final_recon_loss']:.4f}")
            print(f"   Final KL loss: {result['final_kl_loss']:.4f}")
        except Exception as e:
            print(f"   Failed: {e}")
    
    # Save results
    output_dir = Path("outputs/vae_configurations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize
    plt.figure(figsize=(20, 15))
    
    # 1. Training curves
    plt.subplot(3, 4, 1)
    for res in results:
        config = res['config']
        label = f"E{config['encoder_hidden']}-D{config['decoder_hidden']}-L{config['latent_dim']}"
        if config['use_latent_cnf']:
            label += "-CNF"
        plt.plot(res['losses'], label=label, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Curves')
    plt.legend(fontsize=8)
    plt.grid(True)
    
    # 2. Reconstruction loss
    plt.subplot(3, 4, 2)
    for res in results:
        config = res['config']
        label = f"L{config['latent_dim']}"
        plt.plot(res['recon_losses'], label=label, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Quality')
    plt.legend()
    plt.grid(True)
    
    # 3. KL loss
    plt.subplot(3, 4, 3)
    for res in results:
        plt.plot(res['kl_losses'], alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('KL Loss')
    plt.title('KL Divergence')
    plt.grid(True)
    
    # 4. Parameters vs Performance
    plt.subplot(3, 4, 4)
    params = [r['total_params'] for r in results]
    recon_losses = [r['final_recon_loss'] for r in results]
    for i, res in enumerate(results):
        config = res['config']
        plt.scatter(params[i], recon_losses[i], s=200, alpha=0.7)
        plt.annotate(f"L{config['latent_dim']}", (params[i], recon_losses[i]))
    plt.xlabel('Total Parameters')
    plt.ylabel('Final Reconstruction Loss')
    plt.title('Model Size vs Quality')
    plt.xscale('log')
    plt.grid(True)
    
    # 5. Latent dimension effect
    plt.subplot(3, 4, 5)
    latent_dims = [r['config']['latent_dim'] for r in results if not r['config']['use_latent_cnf']]
    final_recons = [r['final_recon_loss'] for r in results if not r['config']['use_latent_cnf']]
    if latent_dims:
        plt.plot(latent_dims, final_recons, 'o-', markersize=10)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Final Reconstruction Loss')
        plt.title('Information Bottleneck Effect')
        plt.grid(True)
    
    # 6. Latent space statistics
    plt.subplot(3, 4, 6)
    for i, res in enumerate(results):
        plt.scatter(res['latent_mean'], res['latent_std'], s=200, alpha=0.7)
        plt.annotate(f"L{res['config']['latent_dim']}", 
                    (res['latent_mean'], res['latent_std']))
    plt.xlabel('Latent Mean')
    plt.ylabel('Latent Std')
    plt.title('Latent Space Statistics')
    plt.grid(True)
    
    # 7. With vs Without Latent CNF
    plt.subplot(3, 4, 7)
    baseline = next((r for r in results if not r['config']['use_latent_cnf'] and 
                    r['config']['latent_dim'] == 32), None)
    with_cnf = next((r for r in results if r['config']['use_latent_cnf'] and 
                    r['config']['latent_dim'] == 32), None)
    if baseline and with_cnf:
        labels = ['Without CNF', 'With CNF']
        recon = [baseline['final_recon_loss'], with_cnf['final_recon_loss']]
        kl = [baseline['final_kl_loss'], with_cnf['final_kl_loss']]
        
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, recon, width, label='Recon Loss')
        plt.bar(x + width/2, kl, width, label='KL Loss')
        plt.xticks(x, labels)
        plt.ylabel('Loss')
        plt.title('Effect of Latent CNF')
        plt.legend()
    
    # 8-12. Individual configuration visualizations
    for idx, res in enumerate(results[:5]):
        plt.subplot(3, 4, 8 + idx)
        config = res['config']
        
        # Show loss components
        epochs = range(len(res['losses']))
        plt.plot(epochs[::10], res['recon_losses'][::10], 'o-', label='Recon', markersize=4)
        plt.plot(epochs[::10], res['kl_losses'][::10], 's-', label='KL', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"E{config['encoder_hidden']}-D{config['decoder_hidden']}-L{config['latent_dim']}")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "vae_configuration_results.png", dpi=150)
    plt.close()
    
    # Summary
    print("\n📊 VAE CONFIGURATION SUMMARY")
    print("=" * 50)
    
    # Sort by reconstruction quality
    sorted_results = sorted(results, key=lambda x: x['final_recon_loss'])
    
    print("\n| Config | Encoder | Decoder | Latent | CNF | Params | Recon Loss |")
    print("|--------|---------|---------|--------|-----|--------|------------|")
    for i, res in enumerate(sorted_results):
        config = res['config']
        print(f"| {i+1:6} | {config['encoder_hidden']:7} | {config['decoder_hidden']:7} | "
              f"{config['latent_dim']:6} | {'Yes' if config['use_latent_cnf'] else 'No':3} | "
              f"{res['total_params']:6,} | {res['final_recon_loss']:.4f} |")
    
    # Best configuration
    best = sorted_results[0]
    print(f"\n🏆 Best configuration:")
    print(f"   Encoder hidden: {best['config']['encoder_hidden']}")
    print(f"   Decoder hidden: {best['config']['decoder_hidden']}")
    print(f"   Latent dim: {best['config']['latent_dim']}")
    print(f"   Use Latent CNF: {best['config']['use_latent_cnf']}")
    print(f"   Total parameters: {best['total_params']:,}")
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    
    # Latent dimension insight
    latent_results = [(r['config']['latent_dim'], r['final_recon_loss']) 
                     for r in results if not r['config']['use_latent_cnf']]
    if len(latent_results) > 2:
        best_latent = min(latent_results, key=lambda x: x[1])
        print(f"- Optimal latent dimension: {best_latent[0]}")
    
    # Latent CNF insight
    if baseline and with_cnf:
        improvement = (baseline['final_recon_loss'] - with_cnf['final_recon_loss']) / baseline['final_recon_loss']
        if improvement > 0:
            print(f"- Latent CNF improves reconstruction by {improvement*100:.1f}%")
        else:
            print(f"- Latent CNF not needed (adds {-improvement*100:.1f}% error)")
    
    # Save detailed results
    detailed_results = []
    for res in results:
        detailed_results.append({
            'config': res['config'],
            'total_params': res['total_params'],
            'final_recon_loss': res['final_recon_loss'],
            'final_kl_loss': res['final_kl_loss'],
            'best_recon_loss': res['best_recon_loss']
        })
    
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_dir}/detailed_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_vae_configuration_test(args.data_path, args.slice_name, args.device)
