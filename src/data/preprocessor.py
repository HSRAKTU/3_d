"""
Data preprocessing utilities for PointFlow2D.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SlicePreprocessor:
    """Handles normalization and preprocessing of slice data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def normalize_slice(self, slice_data: np.ndarray, 
                       bbox_min: Optional[Tuple[float, float]] = None,
                       bbox_max: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, dict]:
        """
        Normalize slice coordinates to [-1, 1] range.
        
        Args:
            slice_data: (N, 2) array of points
            bbox_min: Optional custom bbox minimum (y, z)
            bbox_max: Optional custom bbox maximum (y, z)
            
        Returns:
            Tuple of (normalized_points, normalization_info)
        """
        if len(slice_data) == 0:
            return slice_data, {"bbox_min": None, "bbox_max": None, "scale": None, "offset": None}
        
        # Calculate bounding box
        if bbox_min is None or bbox_max is None:
            actual_min = slice_data.min(axis=0)
            actual_max = slice_data.max(axis=0)
        else:
            actual_min = np.array(bbox_min)
            actual_max = np.array(bbox_max)
        
        # Avoid division by zero
        span = actual_max - actual_min
        span = np.where(span == 0, 1.0, span)
        
        # Normalize to [-1, 1]
        center = (actual_max + actual_min) / 2
        scale = span / 2
        
        normalized = (slice_data - center) / scale
        
        normalization_info = {
            "bbox_min": tuple(actual_min),
            "bbox_max": tuple(actual_max),
            "center": tuple(center),
            "scale": tuple(scale)
        }
        
        return normalized, normalization_info
    
    def denormalize_slice(self, normalized_data: np.ndarray, 
                         normalization_info: dict) -> np.ndarray:
        """
        Denormalize slice coordinates back to original range.
        
        Args:
            normalized_data: (N, 2) array of normalized points
            normalization_info: Dictionary from normalize_slice
            
        Returns:
            Denormalized points
        """
        if len(normalized_data) == 0 or normalization_info["scale"] is None:
            return normalized_data
        
        center = np.array(normalization_info["center"])
        scale = np.array(normalization_info["scale"])
        
        return normalized_data * scale + center


def main():
    """CLI interface for testing the SlicePreprocessor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SlicePreprocessor")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Testing SlicePreprocessor...")
        
        preprocessor = SlicePreprocessor()
        
        # Test data
        test_points = np.array([[0.0, 0.0], [2.0, 4.0], [1.0, 2.0]])
        print(f"Original points:\n{test_points}")
        
        # Normalize
        normalized, info = preprocessor.normalize_slice(test_points)
        print(f"Normalized points:\n{normalized}")
        print(f"Normalization info: {info}")
        
        # Denormalize
        denormalized = preprocessor.denormalize_slice(normalized, info)
        print(f"Denormalized points:\n{denormalized}")
        
        # Check if denormalization is correct
        diff = np.abs(test_points - denormalized).max()
        print(f"Max difference after round-trip: {diff}")
        
        if diff < 1e-10:
            print("✅ Round-trip test passed!")
        else:
            print("❌ Round-trip test failed!")
    
    return 0


if __name__ == "__main__":
    exit(main())
