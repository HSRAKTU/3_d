"""
PyTorch Dataset for slice data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data.loader import SliceDataLoader
from data.preprocessor import SlicePreprocessor

logger = logging.getLogger(__name__)


class SliceDataset(Dataset):
    """
    PyTorch Dataset for 2D slice data.
    
    Handles variable-length slices with normalization and optional padding.
    """
    
    def __init__(self, 
                 data_directory: str,
                 car_ids: Optional[List[str]] = None,
                 normalize: bool = True,
                 max_points: Optional[int] = None,
                 min_points: int = 10):
        """
        Initialize the dataset.
        
        Args:
            data_directory: Path to directory with slice files
            car_ids: Optional list of specific car IDs to use
            normalize: Whether to normalize coordinates to [-1, 1]
            max_points: Maximum points per slice (for filtering)
            min_points: Minimum points per slice (for filtering)
        """
        self.data_directory = Path(data_directory)
        self.normalize = normalize
        self.max_points = max_points
        self.min_points = min_points
        
        # Initialize data loader and preprocessor
        self.loader = SliceDataLoader(data_directory)
        self.preprocessor = SlicePreprocessor()
        
        # Get car IDs
        if car_ids is None:
            available_car_ids = self.loader.get_car_ids()
        else:
            available_car_ids = car_ids
        
        # Load and prepare all slice data
        self.slices_data = []
        self._load_all_slices(available_car_ids)
        
        logger.info(f"Loaded {len(self.slices_data)} slices from {len(available_car_ids)} cars")
    
    def _load_all_slices(self, car_ids: List[str]) -> None:
        """Load all slices and prepare dataset entries."""
        for car_id in car_ids:
            try:
                slices = self.loader.load_car_slices(car_id)
                
                for slice_idx, slice_points in enumerate(slices):
                    # Filter by point count
                    if len(slice_points) < self.min_points:
                        continue
                    
                    if self.max_points and len(slice_points) > self.max_points:
                        continue
                    
                    # Store slice info
                    self.slices_data.append({
                        'car_id': car_id,
                        'slice_idx': slice_idx,
                        'points': slice_points,
                        'num_points': len(slice_points)
                    })
                    
            except Exception as e:
                logger.error(f"Failed to load {car_id}: {e}")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.slices_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single slice.
        
        Args:
            idx: Index of the slice
            
        Returns:
            Dictionary containing:
                - 'points': Normalized point coordinates (num_points, 2)
                - 'num_points': Number of points (scalar)
                - 'normalization_info': Normalization parameters
                - 'car_id': Car identifier (string)
                - 'slice_idx': Slice index within car
        """
        slice_info = self.slices_data[idx]
        points = slice_info['points'].copy()
        
        # Normalize if requested
        if self.normalize:
            normalized_points, norm_info = self.preprocessor.normalize_slice(points)
            points = normalized_points
        else:
            norm_info = {'bbox_min': None, 'bbox_max': None, 'scale': None, 'offset': None}
        
        return {
            'points': torch.tensor(points, dtype=torch.float32),
            'num_points': torch.tensor(slice_info['num_points'], dtype=torch.long),
            'normalization_info': norm_info,
            'car_id': slice_info['car_id'],
            'slice_idx': slice_info['slice_idx']
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        point_counts = [item['num_points'] for item in self.slices_data]
        
        return {
            'total_slices': len(self.slices_data),
            'min_points': min(point_counts),
            'max_points': max(point_counts),
            'mean_points': np.mean(point_counts),
            'std_points': np.std(point_counts),
            'unique_cars': len(set(item['car_id'] for item in self.slices_data))
        }


def collate_variable_length(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length slices.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched data with padding and masks
    """
    batch_size = len(batch)
    
    # Find maximum number of points in this batch
    max_points = max(item['num_points'].item() for item in batch)
    
    # Create padded tensors
    padded_points = torch.zeros(batch_size, max_points, 2)
    point_masks = torch.zeros(batch_size, max_points)
    num_points = torch.zeros(batch_size, dtype=torch.long)
    
    # Collect metadata
    car_ids = []
    slice_indices = []
    normalization_infos = []
    
    for i, item in enumerate(batch):
        points = item['points']
        n_points = item['num_points'].item()
        
        # Fill padded tensor
        padded_points[i, :n_points] = points
        point_masks[i, :n_points] = 1.0
        num_points[i] = n_points
        
        # Collect metadata
        car_ids.append(item['car_id'])
        slice_indices.append(item['slice_idx'])
        normalization_infos.append(item['normalization_info'])
    
    return {
        'points': padded_points,
        'mask': point_masks,
        'num_points': num_points,
        'car_ids': car_ids,
        'slice_indices': slice_indices,
        'normalization_infos': normalization_infos
    }


def main():
    """CLI interface for testing the dataset."""
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description="Test SliceDataset")
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-points", type=int, help="Maximum points per slice")
    parser.add_argument("--min-points", type=int, default=10, help="Minimum points per slice")
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to test")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print(f"🧪 Testing SliceDataset...")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Point range: {args.min_points} - {args.max_points or 'unlimited'}")
    
    try:
        # Create dataset
        dataset = SliceDataset(
            data_directory=args.data_dir,
            max_points=args.max_points,
            min_points=args.min_points,
            normalize=True
        )
        
        # Print statistics
        stats = dataset.get_statistics()
        print(f"\n📊 Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.1f}")
            else:
                print(f"   {key}: {value}")
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_variable_length
        )
        
        print(f"\n🔍 Testing data loading...")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            
            print(f"\n   Batch {batch_idx + 1}:")
            print(f"     Points shape: {batch['points'].shape}")
            print(f"     Mask shape: {batch['mask'].shape}")
            print(f"     Num points: {batch['num_points'].tolist()}")
            print(f"     Car IDs: {batch['car_ids']}")
            
            # Verify data integrity
            points = batch['points']
            mask = batch['mask']
            
            # Check that points are within normalized range
            masked_points = points[mask.bool()]
            if len(masked_points) > 0:
                point_range = (masked_points.min().item(), masked_points.max().item())
                print(f"     Point range: [{point_range[0]:.3f}, {point_range[1]:.3f}]")
            
            # Check mask consistency
            expected_mask_sum = batch['num_points'].sum().item()
            actual_mask_sum = mask.sum().item()
            if expected_mask_sum != actual_mask_sum:
                print(f"     ⚠️ Mask inconsistency: expected {expected_mask_sum}, got {actual_mask_sum}")
            else:
                print(f"     ✅ Mask consistency check passed")
        
        print(f"\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
