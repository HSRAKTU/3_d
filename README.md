# PointFlow2D VAE - 2D Point Cloud Generation

A Variational Autoencoder using Continuous Normalizing Flows for generating 2D automotive slice point clouds.

## 🎯 What This Module Does

**Core Capability**: Learn to encode and decode 2D point clouds using a VAE with CNF decoder
- **Input**: Variable-length 2D point clouds (automotive cross-sections)
- **Output**: Latent representations + generated point clouds
- **Architecture**: PointNet2D Encoder + PointFlow CNF Decoder

## 📁 Project Structure

```
src/
├── models/
│   ├── encoder.py              # PointNet2D encoder with attention
│   ├── pointflow_cnf.py        # PointFlow CNF implementation  
│   └── pointflow2d_fixed.py    # Complete VAE model
├── training/
│   ├── dataset.py              # Data loading and batching
│   └── trainer.py              # Training utilities
├── data/
│   ├── loader.py               # Slice file loading
│   ├── analyzer.py             # Data statistics
│   └── preprocessor.py         # Data preprocessing
└── utils/
    └── visualization.py        # Plotting and analysis tools

scripts/
├── prepare_data.py             # Dataset preparation
├── train_full_reconstruction.py # Main training
├── test_overfitting.py         # Single slice validation
├── load_checkpoint.py          # Model loading utilities
└── visualize_pointflow2d.py    # Results visualization
```

## 🚀 Quick Start

### 1. Data Preparation
```bash
# Discover and prepare training dataset
python scripts/prepare_data.py discover data/source_dataset \
  --output-dir data/training_dataset --cars 10

# Validate data quality  
python scripts/prepare_data.py analyze data/training_dataset
```

### 2. Training
```bash
# Basic training
python scripts/train_full_reconstruction.py data/training_dataset

# Full configuration
python scripts/train_full_reconstruction.py data/training_dataset \
  --epochs 50 --batch-size 4 --latent-dim 32 --lr 5e-5 --seed 42
```

### 3. Model Inspection
```bash
# List all checkpoints with performance metrics
python scripts/load_checkpoint.py list --save-dir outputs/full_reconstruction

# Inspect specific checkpoint
python scripts/load_checkpoint.py inspect --checkpoint outputs/checkpoint_epoch_050.pt
```

## 📋 Complete CLI Reference

### `scripts/prepare_data.py`
**Purpose**: Dataset discovery, validation, analysis, and preparation

#### Commands:
```bash
# Discover available data files
python scripts/prepare_data.py discover <source_dir> [options]

# Validate data integrity  
python scripts/prepare_data.py validate <source_dir> [options]

# Analyze dataset statistics
python scripts/prepare_data.py analyze <source_dir> [options]

# Copy selected cars for training
python scripts/prepare_data.py copy <source_dir> <output_dir> [options]
```

#### Arguments:
- `source_dir`: Directory containing .npy slice files
- `output_dir`: Where to copy selected files (for copy command)

#### Options:
- `--cars N`: Number of cars to select (default: 10)
- `--validate-limit N`: Limit validation to N files (default: 50, use 'all' for everything)
- `--analyze-limit N`: Limit analysis to N files (default: 50)
- `--output-dir DIR`: Output directory for prepared dataset
- `--force`: Overwrite existing output directory

---

### `scripts/train_full_reconstruction.py`
**Purpose**: Main training script for PointFlow2D VAE

#### Usage:
```bash
python scripts/train_full_reconstruction.py <data_dir> [options]
```

#### Required Arguments:
- `data_dir`: Directory containing training data (.npy files)

#### Training Configuration:
- `--epochs N`: Number of training epochs (default: 100)
- `--batch-size N`: Batch size (default: 4)
- `--lr FLOAT`: Learning rate (default: 5e-5)
- `--seed N`: Random seed for reproducibility (default: 42)

#### Model Architecture:
- `--latent-dim N`: Latent space dimension (default: 32)
- `--cnf-hidden N`: CNF hidden layer dimension (default: 64)

#### Loss Configuration:
- `--beta-schedule SCHEDULE`: KL annealing schedule (linear/cosine/constant, default: linear)
- `--beta-start FLOAT`: Starting beta value (default: 0.0)
- `--beta-end FLOAT`: Ending beta value (default: 0.01)

#### Saving Options:
- `--save-every N`: Save checkpoint every N epochs (default: 1)
- `--save-dir DIR`: Directory to save outputs (default: outputs/full_reconstruction)

#### Example Configurations:
```bash
# Small model (55K params) - for testing
python scripts/train_full_reconstruction.py data/training_dataset \
  --latent-dim 32 --cnf-hidden 64 --batch-size 4

# Medium model (200K params) - for production  
python scripts/train_full_reconstruction.py data/training_dataset \
  --latent-dim 64 --cnf-hidden 128 --batch-size 8 --lr 1e-4

# Large model (500K+ params) - for high-end GPU
python scripts/train_full_reconstruction.py data/training_dataset \
  --latent-dim 128 --cnf-hidden 256 --batch-size 16 --lr 1e-4
```

---

### `scripts/test_overfitting.py`
**Purpose**: Validate model architecture on single slice

#### Usage:
```bash
python scripts/test_overfitting.py <slice_path> [options]
```

#### Required Arguments:
- `slice_path`: Path to single .npy slice file

#### Options:
- `--epochs N`: Number of epochs (default: 200)
- `--latent-dim N`: Latent dimension (default: 32)
- `--cnf-hidden N`: CNF hidden dimension (default: 64)
- `--lr FLOAT`: Learning rate (default: 5e-5)
- `--beta-max FLOAT`: Maximum beta value (default: 0.001)
- `--seed N`: Random seed (default: 42)

#### Example:
```bash
python scripts/test_overfitting.py data/single_slice.npy --epochs 100
```

---

### `scripts/load_checkpoint.py`
**Purpose**: Checkpoint inspection and model loading

#### Commands:
```bash
# List all available checkpoints with metrics
python scripts/load_checkpoint.py list [options]

# Inspect specific checkpoint details
python scripts/load_checkpoint.py inspect --checkpoint <path> 

# Load model for inference
python scripts/load_checkpoint.py load --checkpoint <path> [options]
```

#### Options:
- `--save-dir DIR`: Directory containing checkpoints (default: outputs/full_reconstruction)
- `--checkpoint PATH`: Specific checkpoint file to inspect/load
- `--device DEVICE`: Device for loading (auto/cpu/cuda, default: auto)

---

### `scripts/visualize_pointflow2d.py`
**Purpose**: Generate visualization plots from trained model

#### Usage:
```bash
python scripts/visualize_pointflow2d.py <checkpoint_path> <data_dir> [options]
```

#### Required Arguments:
- `checkpoint_path`: Path to model checkpoint (.pt file)
- `data_dir`: Directory containing test data

#### Options:
- `--output-dir DIR`: Where to save plots (default: outputs/visualizations)
- `--num-samples N`: Number of samples to generate (default: 16)
- `--device DEVICE`: Device to use (auto/cpu/cuda, default: auto)

## 📊 Understanding Outputs

### Training Logs
- **Reconstruction Loss**: Negative log-likelihood (higher = better, target: < -10)
- **KL Loss**: Latent space regularization (lower = better, target: < 100)  
- **Beta**: KL weighting factor (gradually increases during training)

### Saved Files
- `checkpoint_epoch_XXX.pt`: Model weights and training state
- `training_metadata.json`: Complete training configuration and progress
- `training_losses.json`: Loss curves for plotting

### Success Criteria
- **Architecture Validation**: Single slice overfitting test succeeds
- **Reconstruction Quality**: Loss improves from ~2.0 to < -5.0
- **Latent Stability**: KL loss stabilizes below 100
- **Generation Quality**: Samples look like reasonable automotive slices

## 🔧 Development Workflow

1. **Data Prep**: `prepare_data.py discover` → `copy` → `analyze`
2. **Architecture Test**: `test_overfitting.py` on single slice
3. **Full Training**: `train_full_reconstruction.py` with desired config
4. **Monitor Progress**: Check logs and `load_checkpoint.py list`
5. **Analyze Results**: `visualize_pointflow2d.py` for plots and insights

## ⚙️ Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slow but functional)
- **Recommended**: GPU with 6GB+ VRAM for reasonable speed
- **Optimal**: Modern GPU (RTX 3080+) for large models and fast iteration
