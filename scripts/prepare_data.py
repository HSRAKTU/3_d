#!/usr/bin/env python3
"""
Data preparation script for PointFlow2D.

This script discovers, analyzes, and prepares slice data for training.
Includes progress bars, validation, and comprehensive logging.
"""

import os
import sys
import shutil
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import SliceDataLoader
from data.analyzer import SliceAnalyzer


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    # Create logs directory
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / 'data_preparation.log')
        ]
    )


def discover_and_validate_data(source_dir: str, validate_limit: str = "10") -> SliceDataLoader:
    """
    Discover and validate slice data in source directory.
    
    Args:
        source_dir: Path to source data directory
        validate_limit: Number of files to validate ("all" for complete validation)
        
    Returns:
        Validated SliceDataLoader instance
    """
    print("🔍 Discovering slice data...")
    loader = SliceDataLoader(source_dir)
    
    if loader.get_file_count() == 0:
        raise FileNotFoundError(f"No slice files found in {source_dir}")
    
    print(f"📁 Found {loader.get_file_count()} slice files")
    
    # Validate data with limit
    if validate_limit.lower() == "all":
        print("✅ Validating ALL data files integrity...")
        car_ids_to_validate = loader.get_car_ids()
    else:
        try:
            limit = int(validate_limit)
            print(f"✅ Validating {limit} data files integrity...")
            car_ids_to_validate = loader.get_car_ids()[:limit]
        except ValueError:
            raise ValueError(f"Invalid validate-limit: {validate_limit}. Use a number or 'all'")
    
    # Custom validation with progress bar
    errors = []
    with tqdm(car_ids_to_validate, desc="Validating", unit="file") as pbar:
        for car_id in pbar:
            try:
                slices = loader.load_car_slices(car_id)
                
                # Additional validation
                if len(slices) == 0:
                    errors.append(f"{car_id}: No slices found")
                    continue
                
                non_empty_slices = sum(1 for s in slices if len(s) > 0)
                if non_empty_slices == 0:
                    errors.append(f"{car_id}: All slices are empty")
                
                pbar.set_postfix({"current": car_id[:15]})
                
            except Exception as e:
                errors.append(f"{car_id}: {e}")
    
    if errors:
        print(f"❌ Found {len(errors)} validation errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
        raise ValueError("Data validation failed")
    
    print(f"✅ Validated {len(car_ids_to_validate)} files successfully")
    return loader


def analyze_dataset(loader: SliceDataLoader, max_cars_to_analyze: int = 100) -> Dict:
    """
    Perform comprehensive dataset analysis.
    
    Args:
        loader: SliceDataLoader instance
        max_cars_to_analyze: Maximum number of cars to analyze (for efficiency)
        
    Returns:
        Analysis results dictionary
    """
    print("📊 Analyzing dataset...")
    analyzer = SliceAnalyzer(loader)
    
    # Sample cars for analysis (much more efficient)
    all_car_ids = loader.get_car_ids()
    if len(all_car_ids) > max_cars_to_analyze:
        print(f"🎯 Sampling {max_cars_to_analyze} cars from {len(all_car_ids)} total for analysis")
        # Use numpy for consistent sampling
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(all_car_ids), max_cars_to_analyze, replace=False)
        car_ids_to_analyze = [all_car_ids[i] for i in sorted(sample_indices)]
    else:
        car_ids_to_analyze = all_car_ids
        print(f"📊 Analyzing all {len(car_ids_to_analyze)} cars")
    
    # Analyze sampled cars with progress bar
    car_stats = {}
    
    with tqdm(car_ids_to_analyze, desc="Analyzing cars", unit="car") as pbar:
        for car_id in pbar:
            try:
                car_stats[car_id] = analyzer.analyze_car(car_id)
                pbar.set_postfix({"current": car_id[:15]})
            except Exception as e:
                logging.error(f"Failed to analyze {car_id}: {e}")
    
    # Create a lightweight summary from the analyzed subset
    if car_stats:
        total_points_analyzed = sum(stats.total_points for stats in car_stats.values())
        total_slices_analyzed = sum(stats.num_slices for stats in car_stats.values())
        avg_points_per_slice_analyzed = [stats.avg_points_per_slice for stats in car_stats.values() if stats.avg_points_per_slice > 0]
        
        summary = {
            "total_cars": len(all_car_ids),  # Total in dataset
            "analyzed_cars": len(car_stats),  # Actually analyzed
            "total_slices_analyzed": total_slices_analyzed,
            "total_points_analyzed": total_points_analyzed,
            "avg_slices_per_car": total_slices_analyzed / len(car_stats),
            "avg_points_per_slice": np.mean(avg_points_per_slice_analyzed) if avg_points_per_slice_analyzed else 0,
            "car_points_min": min(stats.total_points for stats in car_stats.values()),
            "car_points_max": max(stats.total_points for stats in car_stats.values()),
            "car_points_std": np.std([stats.total_points for stats in car_stats.values()]),
            "valid_cars": len([s for s in car_stats.values() if s.total_points > 0]),
        }
    else:
        summary = {"error": "No cars analyzed successfully"}
    
    return {
        "summary": summary,
        "car_stats": car_stats,
        "analyzer": analyzer
    }


def select_working_dataset(analyzer: SliceAnalyzer, analyzed_stats: Dict, num_cars: int, strategy: str) -> List[str]:
    """
    Select cars for the working dataset.
    
    Args:
        analyzer: SliceAnalyzer instance
        analyzed_stats: Dictionary of analyzed car statistics
        num_cars: Number of cars to select
        strategy: Selection strategy
        
    Returns:
        List of selected car IDs
    """
    print(f"🎯 Selecting {num_cars} cars using '{strategy}' strategy from analyzed subset...")
    
    # Use only the analyzed cars for selection to be efficient
    valid_cars = [(car_id, stats) for car_id, stats in analyzed_stats.items() if stats.total_points > 0]
    
    if len(valid_cars) <= num_cars:
        selected_cars = [car_id for car_id, _ in valid_cars]
    else:
        if strategy == "diverse":
            # Sort by total points and select evenly spaced
            sorted_cars = sorted(valid_cars, key=lambda x: x[1].total_points)
            step = len(sorted_cars) // num_cars
            selected_cars = [sorted_cars[i * step][0] for i in range(num_cars)]
            
        elif strategy == "random":
            np.random.shuffle(valid_cars)
            selected_cars = [car_id for car_id, _ in valid_cars[:num_cars]]
            
        elif strategy == "largest":
            sorted_cars = sorted(valid_cars, key=lambda x: x[1].total_points, reverse=True)
            selected_cars = [car_id for car_id, _ in sorted_cars[:num_cars]]
            
        elif strategy == "smallest":
            sorted_cars = sorted(valid_cars, key=lambda x: x[1].total_points)
            selected_cars = [car_id for car_id, _ in sorted_cars[:num_cars]]
            
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    print("✅ Selected cars:")
    for i, car_id in enumerate(selected_cars):
        print(f"   {i+1:2d}. {car_id}")
    
    return selected_cars


def copy_working_dataset(loader: SliceDataLoader, selected_cars: List[str], target_dir: str) -> None:
    """
    Copy selected cars to the working dataset directory.
    
    Args:
        loader: SliceDataLoader instance
        selected_cars: List of car IDs to copy
        target_dir: Target directory path
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Copying {len(selected_cars)} cars to {target_dir}...")
    
    copied_files = []
    with tqdm(selected_cars, desc="Copying files", unit="file") as pbar:
        for car_id in pbar:
            source_file = loader.data_directory / f"{car_id}_axis-x.npy"
            target_file = target_path / f"{car_id}_axis-x.npy"
            
            try:
                shutil.copy2(source_file, target_file)
                copied_files.append(str(target_file))
                pbar.set_postfix({"current": car_id[:15]})
            except Exception as e:
                logging.error(f"Failed to copy {car_id}: {e}")
    
    print(f"✅ Successfully copied {len(copied_files)} files")
    return copied_files


def save_metadata(analysis_results: Dict, selected_cars: List[str], target_dir: str, 
                 source_dir: str, strategy: str) -> None:
    """
    Save dataset metadata and analysis results.
    
    Args:
        analysis_results: Analysis results from analyze_dataset
        selected_cars: List of selected car IDs
        target_dir: Target directory path
        source_dir: Source directory path
        strategy: Selection strategy used
    """
    metadata = {
        "dataset_info": {
            "name": "PointFlow2D Working Dataset",
            "source_directory": str(source_dir),
            "target_directory": str(target_dir),
            "num_cars": len(selected_cars),
            "selection_strategy": strategy,
            "selected_cars": selected_cars
        },
        "dataset_summary": {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v 
                            for k, v in analysis_results["summary"].items()},
        "car_statistics": {}
    }
    
    # Add statistics for selected cars
    for car_id in selected_cars:
        if car_id in analysis_results["car_stats"]:
            stats = analysis_results["car_stats"][car_id]
            metadata["car_statistics"][car_id] = {
                "num_slices": int(stats.num_slices),
                "total_points": int(stats.total_points),
                "non_empty_slices": int(stats.non_empty_slices),
                "avg_points_per_slice": float(stats.avg_points_per_slice),
                "global_bbox_min": [float(x) for x in stats.global_bbox_min] if stats.global_bbox_min else None,
                "global_bbox_max": [float(x) for x in stats.global_bbox_max] if stats.global_bbox_max else None
            }
    
    # Save metadata
    metadata_path = Path(target_dir) / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata saved to {metadata_path}")


def validate_working_dataset(target_dir: str) -> bool:
    """
    Validate the created working dataset.
    
    Args:
        target_dir: Target directory path
        
    Returns:
        True if validation successful
    """
    print("🔍 Validating working dataset...")
    
    try:
        working_loader = SliceDataLoader(target_dir)
        is_valid, errors = working_loader.validate_data_directory()
        
        if is_valid:
            # Quick analysis
            working_analyzer = SliceAnalyzer(working_loader)
            summary = working_analyzer.get_dataset_summary()
            
            print("✅ Working dataset validation successful!")
            print(f"📊 Dataset summary:")
            print(f"   Cars: {summary['total_cars']}")
            print(f"   Total slices: {summary['total_slices']:,}")
            print(f"   Total points: {summary['total_points']:,}")
            print(f"   Avg points per slice: {summary['avg_points_per_slice']:.1f}")
            
            return True
        else:
            print(f"❌ Validation failed with {len(errors)} errors")
            for error in errors[:5]:
                print(f"   - {error}")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    """Main data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare slice data for PointFlow2D training")
    parser.add_argument("source_dir", help="Source directory with slice .npy files")
    parser.add_argument("--target-dir", default="data/selected_slices", 
                       help="Target directory for working dataset")
    parser.add_argument("--num-cars", type=int, default=5,
                       help="Number of cars to select for working dataset")
    parser.add_argument("--strategy", default="diverse",
                       choices=["diverse", "random", "largest", "smallest"],
                       help="Car selection strategy")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing target directory")
    parser.add_argument("--validate-limit", default="10",
                       help="Number of files to validate (use 'all' for complete validation)")
    parser.add_argument("--analyze-limit", type=int, default=50,
                       help="Maximum number of cars to analyze for selection")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Check if target directory exists
        target_path = Path(args.target_dir)
        if target_path.exists() and not args.force:
            print(f"❌ Target directory {args.target_dir} already exists. Use --force to overwrite.")
            return 1
        
        print("🚀 Starting PointFlow2D data preparation...")
        print(f"📁 Source: {args.source_dir}")
        print(f"📁 Target: {args.target_dir}")
        print(f"🎯 Cars to select: {args.num_cars}")
        print(f"📋 Strategy: {args.strategy}")
        
        # Step 1: Discover and validate source data
        loader = discover_and_validate_data(args.source_dir, args.validate_limit)
        
        # Step 2: Analyze dataset
        analysis_results = analyze_dataset(loader, args.analyze_limit)
        
        # Step 3: Select working dataset
        selected_cars = select_working_dataset(
            analysis_results["analyzer"], 
            analysis_results["car_stats"],
            args.num_cars, 
            args.strategy
        )
        
        # Step 4: Copy files
        copy_working_dataset(loader, selected_cars, args.target_dir)
        
        # Step 5: Save metadata
        save_metadata(
            analysis_results, 
            selected_cars, 
            args.target_dir, 
            args.source_dir, 
            args.strategy
        )
        
        # Step 6: Validate working dataset
        validation_success = validate_working_dataset(args.target_dir)
        
        if validation_success:
            print("\n🎉 Data preparation completed successfully!")
            print(f"📁 Working dataset: {os.path.abspath(args.target_dir)}")
            print("🚀 Ready for PointFlow2D model training!")
        else:
            print("\n❌ Data preparation completed with validation errors")
            return 1
    
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
