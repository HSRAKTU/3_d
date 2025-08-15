#!/usr/bin/env python3
"""
Quick script to check data format and find a valid slice for testing.
"""

import numpy as np
import torch
from pathlib import Path
import sys

def check_data_format(data_path: str):
    """Check what format the data is in and find a valid slice."""
    
    data_dir = Path(data_path)
    print(f"Checking data directory: {data_dir}")
    
    # Look for numpy files
    npy_files = list(data_dir.glob("*.npy"))
    npz_files = list(data_dir.glob("*.npz"))
    pt_files = list(data_dir.glob("*.pt"))
    pth_files = list(data_dir.glob("*.pth"))
    
    print(f"\nFound files:")
    print(f"  .npy files: {len(npy_files)}")
    print(f"  .npz files: {len(npz_files)}")
    print(f"  .pt files: {len(pt_files)}")
    print(f"  .pth files: {len(pth_files)}")
    
    # Check subdirectories
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"\nSubdirectories: {[d.name for d in subdirs[:5]]}...")
        
        # Check first subdir
        first_subdir = subdirs[0]
        sub_npy = list(first_subdir.glob("*.npy"))
        sub_npz = list(first_subdir.glob("*.npz"))
        print(f"\nIn {first_subdir.name}:")
        print(f"  .npy files: {len(sub_npy)}")
        print(f"  .npz files: {len(sub_npz)}")
        
        # Try loading a file
        if sub_npy:
            test_file = sub_npy[0]
            print(f"\nTesting file: {test_file}")
            try:
                data = np.load(test_file)
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                if len(data.shape) == 2 and data.shape[1] == 2:
                    print(f"  ✓ Looks like 2D points!")
                    print(f"\nSuggested command:")
                    print(f"python scripts/test_decoder_only.py {data_dir} --slice-name {test_file.relative_to(data_dir)}")
            except Exception as e:
                print(f"  Error loading: {e}")
                
    # Try loading single_slice_test.npy if it exists
    single_slice = data_dir / "single_slice_test.npy"
    if single_slice.exists():
        print(f"\nFound single_slice_test.npy!")
        try:
            data = np.load(single_slice)
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_data_format.py <data_directory>")
        sys.exit(1)
    
    check_data_format(sys.argv[1])
