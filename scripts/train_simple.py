#!/usr/bin/env python3
"""
SIMPLIFIED two-stage training - showing it's not complex at all!
This is all you need to get started.
"""

import torch
from pathlib import Path
import argparse
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import PointFlow2DVAE

def train_autoencoder(data_path: str, slice_name: str, device: str = 'cuda'):
    """Dead simple two-stage training."""
    
    # Load your single slice
    data = torch.load(Path(data_path) / slice_name)
    x = data['points'].float().to(device)  # [N, 2]
    mask = data['mask'].float().to(device)  # [N]
    
    # Batch it
    x = x.unsqueeze(0)      # [1, N, 2]
    mask = mask.unsqueeze(0) # [1, N]
    
    print(f"Training on slice with {mask.sum().item()} points")
    
    # Create model with SAFE defaults
    model = PointFlow2DVAE(
        zdim=128,
        encoder_out_dim=256,
        
        # Stage 1 settings (deterministic)
        use_deterministic_encoder=True,
        use_latent_flow=False,
        
        # Safe CNF settings
        cnf_solver='dopri5',
        atol=1e-3,  # Relaxed tolerance
        rtol=1e-3,  # Relaxed tolerance
        
        # Training settings
        lr=5e-5,    # Very safe learning rate
        
        # Point CNF architecture (keeping it robust, not toy!)
        point_dims='512-512-512',  # 3 hidden layers
        point_layer_type='concatsquash',
        point_activation='softplus',
        point_final_activation='tanh',
        
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    # ============ STAGE 1: Deterministic Autoencoder ============
    print("\n🔵 STAGE 1: Training Deterministic Autoencoder")
    print("This just learns: slice → latent → slice")
    
    model.use_deterministic_encoder = True
    model.prior_weight = 0.0
    model.entropy_weight = 0.0
    model.recon_weight = 1.0
    
    stage1_epochs = 500
    for epoch in tqdm(range(stage1_epochs), desc="Stage 1"):
        optimizer.zero_grad()
        
        # Forward pass
        recon_x = model.reconstruct(x, mask=mask)
        
        # Simple reconstruction loss
        diff = (x - recon_x) ** 2
        recon_loss = (diff * mask.unsqueeze(-1)).sum() / mask.sum()
        
        recon_loss.backward()
        
        # Gradient clipping for safety
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Recon Loss = {recon_loss.item():.4f}")
    
    print("✅ Stage 1 Complete! Model learned to reconstruct.")
    
    # ============ STAGE 2: Full VAE ============
    print("\n🟢 STAGE 2: Training Full VAE with Latent Flow")
    print("Now we add: variational encoder + latent flow")
    
    # Switch to VAE mode
    model.use_deterministic_encoder = False
    model.use_latent_flow = True
    model.prior_weight = 0.1     # Start small
    model.entropy_weight = 0.001 # Very small
    model.recon_weight = 1.0
    
    # Reinitialize latent flow (it wasn't trained in stage 1)
    if model.latent_cnf is not None:
        model.latent_cnf.reset_parameters()
    
    # Fresh optimizer for stage 2
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # Even safer
    
    stage2_epochs = 500
    for epoch in tqdm(range(stage2_epochs), desc="Stage 2"):
        optimizer.zero_grad()
        
        # Full VAE forward pass
        out = model(x, mask=mask)
        loss = out['loss']
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Total = {loss.item():.4f}, "
                  f"Recon = {out['recon_loss'].item():.4f}, "
                  f"Prior = {out['prior_loss'].item():.4f}")
    
    print("✅ Stage 2 Complete! Full generative model trained.")
    
    # Save the model
    output_dir = Path("outputs/two_stage_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"\n💾 Model saved to {output_dir}/model.pt")
    
    # Test generation
    print("\n🎨 Testing generation from prior...")
    with torch.no_grad():
        z = torch.randn(1, 128).to(device)  # Random latent
        generated = model.decode(z)
        print(f"Generated {generated.shape[1]} points")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--slice-name', type=str, default='single_slice_test.npy')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    train_autoencoder(args.data_path, args.slice_name, args.device)
