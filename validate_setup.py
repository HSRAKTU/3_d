#!/usr/bin/env python3
"""
Validate that all necessary files are in place for RunPod deployment.
"""

import os
import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists and report status."""
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("🔍 Validating PointFlow2D Setup for RunPod Deployment\n")
    
    all_good = True
    
    # Check core model files
    print("📦 Core Model Files:")
    all_good &= check_file("src/models/__init__.py", "Models package")
    all_good &= check_file("src/models/pointflow2d_final.py", "Main VAE model")
    all_good &= check_file("src/models/encoder.py", "PointNet2D encoder")
    all_good &= check_file("src/models/pointflow_cnf.py", "Point CNF")
    all_good &= check_file("src/models/latent_cnf.py", "Latent CNF")
    all_good &= check_file("src/models/point_count_predictor.py", "Point count predictor (future)")
    
    print("\n📊 Training Infrastructure:")
    all_good &= check_file("src/training/dataset.py", "Dataset loader")
    all_good &= check_file("src/training/trainer.py", "Training loop")
    all_good &= check_file("src/utils/validation.py", "Validation utilities")
    all_good &= check_file("src/utils/visualization.py", "Visualization tools")
    
    print("\n🚀 Scripts:")
    all_good &= check_file("scripts/test_single_slice_overfit.py", "Single slice test")
    all_good &= check_file("scripts/run_experiment.py", "Experiment runner")
    all_good &= check_file("scripts/prepare_data.py", "Data preparation")
    
    print("\n⚙️ Configuration:")
    all_good &= check_file("config/experiments.json", "Experiment configs")
    all_good &= check_file("requirements.txt", "Python dependencies")
    all_good &= check_file("setup_runpod.sh", "RunPod setup script")
    all_good &= check_file("RUNPOD_INSTRUCTIONS.md", "RunPod instructions")
    
    print("\n📂 Data:")
    all_good &= check_file("data/single_slice_test.npy", "Test slice")
    
    # Check if data directories exist
    data_dirs = ["data/processed", "data/source_dataset", "data/working_dataset"]
    for dir_path in data_dirs:
        exists = Path(dir_path).exists()
        if exists:
            print(f"✅ Data directory: {dir_path}")
        else:
            print(f"⚠️  Data directory missing: {dir_path} (optional)")
    
    print("\n" + "="*50)
    if all_good:
        print("✅ All essential files are present! Ready for RunPod deployment.")
        print("\n📝 Next steps:")
        print("1. Push code to repository")
        print("2. Clone on RunPod")
        print("3. Run ./setup_runpod.sh")
        print("4. Start training with GPU!")
    else:
        print("❌ Some files are missing. Please check above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
