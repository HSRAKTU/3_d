"""
Unit tests for data loading and analysis modules.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import SliceDataLoader
from data.analyzer import SliceAnalyzer


class TestSliceDataLoader(unittest.TestCase):
    """Test cases for SliceDataLoader."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock slice data
        test_slices = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),  # 2 points
            np.array([[0.5, 0.5]]),              # 1 point
            np.array([]).reshape(0, 2),          # Empty slice
            np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])  # 3 points
        ]
        
        # Save test car
        test_file = self.temp_path / "test_car_001_axis-x.npy"
        np.save(test_file, np.array(test_slices, dtype=object), allow_pickle=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = SliceDataLoader(self.temp_dir)
        self.assertEqual(loader.get_file_count(), 1)
        
        car_ids = loader.get_car_ids()
        self.assertEqual(len(car_ids), 1)
        self.assertEqual(car_ids[0], "test_car_001")
    
    def test_load_car_slices(self):
        """Test loading car slices."""
        loader = SliceDataLoader(self.temp_dir)
        slices = loader.load_car_slices("test_car_001")
        
        self.assertEqual(len(slices), 4)
        self.assertEqual(len(slices[0]), 2)  # First slice has 2 points
        self.assertEqual(len(slices[1]), 1)  # Second slice has 1 point
        self.assertEqual(len(slices[2]), 0)  # Third slice is empty
        self.assertEqual(len(slices[3]), 3)  # Fourth slice has 3 points
    
    def test_validation(self):
        """Test data validation."""
        loader = SliceDataLoader(self.temp_dir)
        is_valid, errors = loader.validate_data_directory()
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)


class TestSliceAnalyzer(unittest.TestCase):
    """Test cases for SliceAnalyzer."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock slice data with known properties
        test_slices = [
            np.array([[0.0, 0.0], [2.0, 2.0]]),  # 2 points, bbox (0,0) to (2,2)
            np.array([[1.0, 1.0]]),              # 1 point at (1,1)
            np.array([]).reshape(0, 2),          # Empty slice
        ]
        
        test_file = self.temp_path / "test_car_002_axis-x.npy"
        np.save(test_file, np.array(test_slices, dtype=object), allow_pickle=True)
        
        self.loader = SliceDataLoader(self.temp_dir)
        self.analyzer = SliceAnalyzer(self.loader)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_slice(self):
        """Test single slice analysis."""
        # Test non-empty slice
        slice_data = np.array([[0.0, 0.0], [2.0, 2.0]])
        stats = self.analyzer.analyze_slice(slice_data, 0)
        
        self.assertEqual(stats.point_count, 2)
        self.assertFalse(stats.is_empty)
        self.assertEqual(stats.bbox_min, (0.0, 0.0))
        self.assertEqual(stats.bbox_max, (2.0, 2.0))
        self.assertEqual(stats.centroid, (1.0, 1.0))
        
        # Test empty slice
        empty_slice = np.array([]).reshape(0, 2)
        empty_stats = self.analyzer.analyze_slice(empty_slice, 1)
        
        self.assertEqual(empty_stats.point_count, 0)
        self.assertTrue(empty_stats.is_empty)
        self.assertIsNone(empty_stats.bbox_min)
    
    def test_analyze_car(self):
        """Test car analysis."""
        stats = self.analyzer.analyze_car("test_car_002")
        
        self.assertEqual(stats.car_id, "test_car_002")
        self.assertEqual(stats.num_slices, 3)
        self.assertEqual(stats.total_points, 3)  # 2 + 1 + 0
        self.assertEqual(stats.empty_slices, 1)
        self.assertEqual(stats.non_empty_slices, 2)
        self.assertEqual(stats.min_points_per_slice, 1)
        self.assertEqual(stats.max_points_per_slice, 2)
    
    def test_dataset_summary(self):
        """Test dataset summary generation."""
        summary = self.analyzer.get_dataset_summary()
        
        self.assertEqual(summary["total_cars"], 1)
        self.assertEqual(summary["total_slices"], 3)
        self.assertEqual(summary["total_points"], 3)
        self.assertEqual(summary["valid_cars"], 1)


def main():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    main()
