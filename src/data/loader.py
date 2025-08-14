"""
Data loading utilities for PointFlow2D slice data.
"""

import os
import glob
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SliceDataLoader:
    """Handles loading and basic validation of slice data files."""
    
    def __init__(self, data_directory: str):
        """
        Initialize the data loader.
        
        Args:
            data_directory: Path to directory containing .npy slice files
        """
        self.data_directory = Path(data_directory)
        self.slice_files = []
        self._discover_files()
    
    def _discover_files(self) -> None:
        """Discover all .npy slice files in the data directory."""
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
        
        pattern = str(self.data_directory / "*_axis-x.npy")
        self.slice_files = sorted(glob.glob(pattern))
        
        if not self.slice_files:
            logger.warning(f"No slice files found in {self.data_directory}")
        else:
            logger.info(f"Found {len(self.slice_files)} slice files")
    
    def get_car_ids(self) -> List[str]:
        """Get list of car IDs from discovered files."""
        car_ids = []
        for file_path in self.slice_files:
            filename = os.path.basename(file_path)
            car_id = filename.replace("_axis-x.npy", "")
            car_ids.append(car_id)
        return car_ids
    
    def load_car_slices(self, car_id: str) -> np.ndarray:
        """
        Load slices for a specific car.
        
        Args:
            car_id: ID of the car to load
            
        Returns:
            Array of slices, each slice is a (N_i, 2) array where N_i varies
            
        Raises:
            FileNotFoundError: If car file doesn't exist
            ValueError: If file is corrupted or invalid
        """
        file_path = self.data_directory / f"{car_id}_axis-x.npy"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Car file not found: {file_path}")
        
        try:
            slices = np.load(file_path, allow_pickle=True)
            
            # Basic validation
            if not isinstance(slices, np.ndarray):
                raise ValueError(f"Invalid slice data format in {file_path}")
            
            # Validate each slice
            for i, slice_data in enumerate(slices):
                if not isinstance(slice_data, np.ndarray):
                    raise ValueError(f"Slice {i} is not a numpy array in {car_id}")
                if slice_data.ndim != 2 and len(slice_data) > 0:
                    raise ValueError(f"Slice {i} has wrong dimensions in {car_id}")
                if len(slice_data) > 0 and slice_data.shape[1] != 2:
                    raise ValueError(f"Slice {i} should have 2 coordinates in {car_id}")
            
            return slices
            
        except Exception as e:
            raise ValueError(f"Failed to load {car_id}: {e}")
    
    def load_all_cars(self) -> Dict[str, np.ndarray]:
        """
        Load slices for all cars.
        
        Returns:
            Dictionary mapping car_id to slice arrays
        """
        all_data = {}
        car_ids = self.get_car_ids()
        
        for car_id in car_ids:
            try:
                all_data[car_id] = self.load_car_slices(car_id)
                logger.debug(f"Loaded {car_id}")
            except Exception as e:
                logger.error(f"Failed to load {car_id}: {e}")
        
        return all_data
    
    def get_file_count(self) -> int:
        """Get number of discovered slice files."""
        return len(self.slice_files)
    
    def validate_data_directory(self) -> Tuple[bool, List[str]]:
        """
        Validate all files in the data directory.
        
        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []
        car_ids = self.get_car_ids()
        
        for car_id in car_ids:
            try:
                slices = self.load_car_slices(car_id)
                
                # Additional validation
                if len(slices) == 0:
                    errors.append(f"{car_id}: No slices found")
                    continue
                
                non_empty_slices = sum(1 for s in slices if len(s) > 0)
                if non_empty_slices == 0:
                    errors.append(f"{car_id}: All slices are empty")
                
            except Exception as e:
                errors.append(f"{car_id}: {e}")
        
        return len(errors) == 0, errors


def main():
    """CLI interface for testing the SliceDataLoader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SliceDataLoader")
    parser.add_argument("data_dir", help="Path to slice data directory")
    parser.add_argument("--car-id", help="Specific car ID to load")
    parser.add_argument("--validate", action="store_true", help="Validate all files")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        loader = SliceDataLoader(args.data_dir)
        
        if args.validate:
            print("🔍 Validating data directory...")
            is_valid, errors = loader.validate_data_directory()
            
            if is_valid:
                print("✅ All files are valid!")
            else:
                print(f"❌ Found {len(errors)} errors:")
                for error in errors:
                    print(f"  - {error}")
            return
        
        if args.car_id:
            print(f"📁 Loading car: {args.car_id}")
            slices = loader.load_car_slices(args.car_id)
            print(f"✅ Loaded {len(slices)} slices")
            
            non_empty = [s for s in slices if len(s) > 0]
            total_points = sum(len(s) for s in non_empty)
            print(f"📊 Non-empty slices: {len(non_empty)}, Total points: {total_points:,}")
        else:
            print(f"📁 Found {loader.get_file_count()} cars")
            car_ids = loader.get_car_ids()
            print("🚗 Available cars:")
            for i, car_id in enumerate(car_ids[:10]):  # Show first 10
                print(f"  {i+1:2d}. {car_id}")
            
            if len(car_ids) > 10:
                print(f"  ... and {len(car_ids) - 10} more")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
