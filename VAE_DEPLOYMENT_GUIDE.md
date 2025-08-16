# PointFlow2D VAE Deployment Guide

## Quick Start

### 1. Test the Implementation
```bash
# Activate environment
cd /workspace/slice_cfd_prototype
source .slice-cfd-proto/bin/activate
cd generative_3d_research

# Run tests
python scripts/test_vae_2d.py
```

### 2. Start Training with Default Settings
```bash
# Basic training (deterministic encoder)
./scripts/launch_vae_training.sh --name baseline --num_cars 10 --epochs 500

# Stochastic VAE training
./scripts/launch_vae_training.sh --name stochastic --num_cars 10 --epochs 500 --stochastic
```

### 3. Custom Training
```bash
python scripts/train_vae_2d.py \
    --batch_size 8 \
    --epochs 1000 \
    --lr 5e-4 \
    --num_cars 20 \
    --solver_steps 10 \
    --lambda_recon 1.0 \
    --lambda_prior 0.1 \
    --lambda_entropy 0.01 \
    --lambda_chamfer 10.0 \
    --lambda_volume 0.01 \
    --output_dir outputs/custom_experiment
```

## Key Features

### 1. Three-Loss VAE Framework
- **Reconstruction Loss**: Ensures accurate point cloud reconstruction
- **Prior Loss**: Regularizes latent space to be Gaussian
- **Entropy Loss**: Prevents posterior collapse (stochastic mode only)

### 2. Additional Regularization
- **Chamfer Loss**: Direct shape similarity metric
- **Volume Regularization**: Prevents log-determinant explosion (negative loss issue)

### 3. Training Modes
- **Deterministic**: Simpler, stable training (recommended to start)
- **Stochastic**: Full VAE with reparameterization trick

## Architecture Details

```
PointFlow2DVAE
├── Encoder: PointNet2D → μ, σ² (latent distribution)
├── Point CNF: Transforms between data ↔ Gaussian
│   ├── Forward: Real slice → Gaussian blob (training)
│   └── Reverse: Gaussian blob → Real slice (generation)
└── Loss computation with proper ELBO
```

## Hyperparameter Guidelines

Based on our experiments:
- **Solver Steps**: 10 (optimal for 2D)
- **Learning Rate**: 5e-4 
- **Batch Size**: 8 (memory efficient)
- **Latent Dim**: 128
- **Hidden Dim**: 256

## Loss Weights (Tunable)
```python
lambda_recon = 1.0      # Reconstruction (primary)
lambda_prior = 0.1      # Latent regularization
lambda_entropy = 0.01   # Diversity (stochastic only)
lambda_chamfer = 10.0   # Shape similarity
lambda_volume = 0.01    # Prevent explosion
```

## Monitoring Training

The script saves:
- `training_log.json`: Loss history
- `best_model.pth`: Best checkpoint by validation Chamfer
- `checkpoint_epoch_N.pth`: Regular checkpoints

Check for:
1. **Loss staying positive** (no cheating)
2. **Chamfer decreasing** (better reconstruction)
3. **Log-det stable** (no explosion)

## Troubleshooting

### High Prior Loss
- Normal at start, should decrease
- If persists, reduce `lambda_prior`

### Negative Total Loss
- Increase `lambda_volume`
- Check log-det values

### Poor Reconstruction
- Increase `lambda_chamfer`
- Try more solver steps (15-20)

### OOM Errors
- Reduce batch size to 4
- Use gradient accumulation

## Next Steps

1. Start with deterministic training
2. Once stable, try stochastic mode
3. Experiment with loss weights
4. Scale to more cars when working

Good luck with your experiments!
