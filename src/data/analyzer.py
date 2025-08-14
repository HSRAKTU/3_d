"""
Slice data analysis utilities for PointFlow2D.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from .loader import SliceDataLoader

logger = logging.getLogger(__name__)


@dataclass
class SliceStats:
    """Statistics for a single slice."""
    slice_index: int
    point_count: int
    bbox_min: Optional[Tuple[float, float]]
    bbox_max: Optional[Tuple[float, float]]
    centroid: Optional[Tuple[float, float]]
    is_empty: bool


@dataclass
class CarStats:
    """Statistics for a car (collection of slices)."""
    car_id: str
    num_slices: int
    total_points: int
    empty_slices: int
    non_empty_slices: int
    min_points_per_slice: int
    max_points_per_slice: int
    avg_points_per_slice: float
    global_bbox_min: Optional[Tuple[float, float]]
    global_bbox_max: Optional[Tuple[float, float]]
    slice_stats: List[SliceStats]


class SliceAnalyzer:
    """Analyzes slice data to extract statistics and insights."""
    
    def __init__(self, data_loader: SliceDataLoader):
        """
        Initialize analyzer with a data loader.
        
        Args:
            data_loader: SliceDataLoader instance
        """
        self.data_loader = data_loader
    
    def analyze_slice(self, slice_data: np.ndarray, slice_index: int) -> SliceStats:
        """
        Analyze a single slice.
        
        Args:
            slice_data: (N, 2) array of points
            slice_index: Index of this slice
            
        Returns:
            SliceStats object with analysis results
        """
        if len(slice_data) == 0:
            return SliceStats(
                slice_index=slice_index,
                point_count=0,
                bbox_min=None,
                bbox_max=None,
                centroid=None,
                is_empty=True
            )
        
        # Calculate bounding box
        bbox_min = tuple(slice_data.min(axis=0))
        bbox_max = tuple(slice_data.max(axis=0))
        
        # Calculate centroid
        centroid = tuple(slice_data.mean(axis=0))
        
        return SliceStats(
            slice_index=slice_index,
            point_count=len(slice_data),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            centroid=centroid,
            is_empty=False
        )
    
    def analyze_car(self, car_id: str) -> CarStats:
        """
        Analyze all slices for a specific car.
        
        Args:
            car_id: ID of the car to analyze
            
        Returns:
            CarStats object with comprehensive analysis
        """
        slices = self.data_loader.load_car_slices(car_id)
        
        # Analyze each slice
        slice_stats = []
        all_points = []
        
        for i, slice_data in enumerate(slices):
            stats = self.analyze_slice(slice_data, i)
            slice_stats.append(stats)
            
            if not stats.is_empty:
                all_points.append(slice_data)
        
        # Calculate car-level statistics
        non_empty_stats = [s for s in slice_stats if not s.is_empty]
        point_counts = [s.point_count for s in non_empty_stats]
        
        if point_counts:
            min_points = min(point_counts)
            max_points = max(point_counts)
            avg_points = np.mean(point_counts)
        else:
            min_points = max_points = avg_points = 0
        
        # Global bounding box
        global_bbox_min = global_bbox_max = None
        if all_points:
            all_points_combined = np.vstack(all_points)
            global_bbox_min = tuple(all_points_combined.min(axis=0))
            global_bbox_max = tuple(all_points_combined.max(axis=0))
        
        return CarStats(
            car_id=car_id,
            num_slices=len(slices),
            total_points=sum(s.point_count for s in slice_stats),
            empty_slices=sum(1 for s in slice_stats if s.is_empty),
            non_empty_slices=len(non_empty_stats),
            min_points_per_slice=min_points,
            max_points_per_slice=max_points,
            avg_points_per_slice=avg_points,
            global_bbox_min=global_bbox_min,
            global_bbox_max=global_bbox_max,
            slice_stats=slice_stats
        )
    
    def analyze_all_cars(self) -> Dict[str, CarStats]:
        """
        Analyze all cars in the dataset.
        
        Returns:
            Dictionary mapping car_id to CarStats
        """
        car_ids = self.data_loader.get_car_ids()
        results = {}
        
        for car_id in car_ids:
            try:
                results[car_id] = self.analyze_car(car_id)
                logger.debug(f"Analyzed {car_id}")
            except Exception as e:
                logger.error(f"Failed to analyze {car_id}: {e}")
        
        return results
    
    def get_dataset_summary(self) -> Dict:
        """
        Get high-level summary statistics for the entire dataset.
        
        Returns:
            Dictionary with dataset-level statistics
        """
        all_stats = self.analyze_all_cars()
        
        if not all_stats:
            return {"error": "No valid cars found"}
        
        # Aggregate statistics
        total_cars = len(all_stats)
        total_slices = sum(stats.num_slices for stats in all_stats.values())
        total_points = sum(stats.total_points for stats in all_stats.values())
        
        # Point distribution statistics
        car_point_counts = [stats.total_points for stats in all_stats.values()]
        avg_points_per_slice_all = [stats.avg_points_per_slice for stats in all_stats.values() if stats.avg_points_per_slice > 0]
        
        return {
            "total_cars": total_cars,
            "total_slices": total_slices,
            "total_points": total_points,
            "avg_slices_per_car": total_slices / total_cars,
            "avg_points_per_car": np.mean(car_point_counts),
            "avg_points_per_slice": np.mean(avg_points_per_slice_all) if avg_points_per_slice_all else 0,
            "car_points_min": min(car_point_counts),
            "car_points_max": max(car_point_counts),
            "car_points_std": np.std(car_point_counts),
            "valid_cars": len([s for s in all_stats.values() if s.total_points > 0]),
        }
    
    def select_representative_cars(self, num_cars: int, strategy: str = "diverse") -> List[str]:
        """
        Select representative cars for training/testing.
        
        Args:
            num_cars: Number of cars to select
            strategy: Selection strategy ("diverse", "random", "largest", "smallest")
            
        Returns:
            List of selected car IDs
        """
        all_stats = self.analyze_all_cars()
        valid_cars = [(car_id, stats) for car_id, stats in all_stats.items() if stats.total_points > 0]
        
        if len(valid_cars) <= num_cars:
            return [car_id for car_id, _ in valid_cars]
        
        if strategy == "diverse":
            # Sort by total points and select evenly spaced
            sorted_cars = sorted(valid_cars, key=lambda x: x[1].total_points)
            step = len(sorted_cars) // num_cars
            selected = [sorted_cars[i * step][0] for i in range(num_cars)]
            
        elif strategy == "random":
            np.random.shuffle(valid_cars)
            selected = [car_id for car_id, _ in valid_cars[:num_cars]]
            
        elif strategy == "largest":
            sorted_cars = sorted(valid_cars, key=lambda x: x[1].total_points, reverse=True)
            selected = [car_id for car_id, _ in sorted_cars[:num_cars]]
            
        elif strategy == "smallest":
            sorted_cars = sorted(valid_cars, key=lambda x: x[1].total_points)
            selected = [car_id for car_id, _ in sorted_cars[:num_cars]]
            
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        return selected


def main():
    """CLI interface for testing the SliceAnalyzer."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Analyze slice data")
    parser.add_argument("data_dir", help="Path to slice data directory")
    parser.add_argument("--car-id", help="Analyze specific car")
    parser.add_argument("--summary", action="store_true", help="Show dataset summary")
    parser.add_argument("--select", type=int, help="Select N representative cars")
    parser.add_argument("--strategy", default="diverse", choices=["diverse", "random", "largest", "smallest"],
                       help="Selection strategy")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        from .loader import SliceDataLoader
        loader = SliceDataLoader(args.data_dir)
        analyzer = SliceAnalyzer(loader)
        
        if args.car_id:
            print(f"🔍 Analyzing car: {args.car_id}")
            stats = analyzer.analyze_car(args.car_id)
            
            print(f"📊 Car Statistics:")
            print(f"   Slices: {stats.num_slices}")
            print(f"   Total points: {stats.total_points:,}")
            print(f"   Non-empty slices: {stats.non_empty_slices}")
            print(f"   Points per slice: {stats.min_points_per_slice} - {stats.max_points_per_slice} (avg: {stats.avg_points_per_slice:.1f})")
            
            if stats.global_bbox_min and stats.global_bbox_max:
                print(f"   Bounding box: {stats.global_bbox_min} to {stats.global_bbox_max}")
        
        elif args.summary:
            print("📊 Analyzing entire dataset...")
            summary = analyzer.get_dataset_summary()
            
            print(f"📋 Dataset Summary:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        
        elif args.select:
            print(f"🎯 Selecting {args.select} representative cars using '{args.strategy}' strategy...")
            selected = analyzer.select_representative_cars(args.select, args.strategy)
            
            print(f"✅ Selected cars:")
            for i, car_id in enumerate(selected):
                print(f"   {i+1:2d}. {car_id}")
            
            if args.output:
                results = {"selected_cars": selected, "strategy": args.strategy}
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Results saved to {args.output}")
        
        else:
            # Default: analyze all cars briefly
            all_stats = analyzer.analyze_all_cars()
            print(f"📊 Analyzed {len(all_stats)} cars")
            
            for car_id, stats in list(all_stats.items())[:5]:  # Show first 5
                print(f"   {car_id}: {stats.total_points:,} points, {stats.non_empty_slices} slices")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
