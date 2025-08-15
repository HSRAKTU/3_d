# Two-Stage Training Strategy

## Why Single-Stage Training Fails

When training all components simultaneously:
- **Encoder** tries to learn meaningful latents
- **Latent CNF** tries to match prior distribution  
- **Point CNF** tries to transform points to Gaussian

This creates instability as all three components interfere with each other's learning.

## The Solution: Two-Stage Training

### Stage 1: Autoencoder Only (Reconstruction)
```python
# Deterministic encoder (no sampling)
z = z_mu  

# No latent flow
log_pz = 0

# Only reconstruction loss
loss = -log_px.mean()
```

**What it learns**: 
- Encoder → meaningful latent codes
- Point CNF → stable point transformation

### Stage 2: Full VAE (Generation)
```python
# Variational encoder (with sampling)
z = z_mu + z_sigma * eps

# Enable latent flow
w, delta_log_pw = latent_cnf(z, ...)

# Full ELBO loss
loss = recon_loss + prior_loss + entropy_loss
```

**What it adds**:
- Latent CNF → prior matching
- Proper variational training

## Usage

```bash
# Run two-stage training
python scripts/train_two_stage.py data/ \
  --stage1-epochs 500 \
  --stage2-epochs 500 \
  --lr-stage1 5e-4 \
  --lr-stage2 1e-4
```

## Why This Works

1. **Stage 1** stabilizes the core transformation without distribution matching complexity
2. **Stage 2** adds generative capability on top of stable reconstruction

This is exactly how the original PointFlow was trained!
