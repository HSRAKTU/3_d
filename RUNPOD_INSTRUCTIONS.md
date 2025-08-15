# RunPod Deployment Instructions for PointFlow2D

## 🚀 Quick Start

1. **Clone the repository** on RunPod:
   ```bash
   git clone <your-repo-url>
   cd slice_cfd_prototype/generative_3d_research
   ```

2. **Run setup script**:
   ```bash
   chmod +x setup_runpod.sh
   ./setup_runpod.sh
   ```

3. **Run single slice overfitting test**:
   ```bash
   python scripts/test_single_slice_overfit.py data/ \
     --slice-name single_slice_test.npy \
     --epochs 1000 \
     --device cuda \
     --viz-freq 50 \
     --save-freq 10
   ```

## 📊 What to Expect

### Success Criteria:
- **Reconstruction loss** should drop below **0.01 nats/dim** (ideally < 0.001)
- **Chamfer distance** should approach **0** (< 0.01 is excellent)
- Visual reconstruction should be nearly perfect

### Output Files:
Results will be saved in `outputs/single_slice_overfit/[timestamp]/`:
- `reconstruction_epoch_XXXX.png` - Visual comparisons
- `checkpoint_epoch_XXXX.pt` - Model checkpoints
- `loss_curves.png` - Training progress

## 🔧 Experiment Configurations

### Available Configs (in `config/experiments.json`):
- `single_slice_overfit` - Validate architecture (209K params)
- `tiny_10cars` - 10-car training (209K params)
- `medium_10cars` - Scaled model (500K params)
- `large_50cars` - Large dataset (1M params)

### To run a specific config:
```bash
python scripts/run_experiment.py --config single_slice_overfit
```

## 🐛 Troubleshooting

### If you see numpy version errors:
```bash
pip install "numpy>=1.21.0,<1.25.0"
```

### To monitor GPU usage:
```bash
nvidia-smi -l 1  # Updates every second
```

### To check training progress:
```bash
tail -f outputs/single_slice_overfit/*/training.log
```

## 📈 Next Steps After Single Slice Success

1. **Scale to 10 cars**:
   ```bash
   python scripts/run_experiment.py --config tiny_10cars
   ```

2. **Test sampling quality**:
   ```bash
   python scripts/test_sampling.py --checkpoint outputs/single_slice_overfit/*/checkpoint_final.pt
   ```

3. **Run validation suite**:
   ```bash
   python scripts/validate_model.py --checkpoint outputs/single_slice_overfit/*/checkpoint_final.pt
   ```

## 💡 Tips for RTX 4090

- The 4090 has 24GB VRAM, so you can use larger batch sizes
- For single slice overfitting, the default settings should work perfectly
- If you encounter OOM errors, reduce `cnf_hidden_dim` or `latent_cnf_hidden_dim`

## 🎯 Key Architecture Features

- **PointNet2D Encoder**: 75,585 parameters
- **Point CNF**: 83,970 parameters  
- **Latent CNF**: 49,600 parameters
- **Total**: 209,155 parameters (for default config)

The model learns to transform automotive slices ↔ Gaussian distributions using invertible normalizing flows!
