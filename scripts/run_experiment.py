#!/usr/bin/env python3
"""
Run experiments using centralized configuration system.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from copy import deepcopy

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from train_with_validation import main as train_main


def load_config(config_path: str, experiment_name: str):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Find experiment in different sections
    experiment_config = None
    section_name = None
    
    for section in ['experiments', 'quick_tests', 'production_configs', 'ablation_studies']:
        if experiment_name in config_data.get(section, {}):
            experiment_config = config_data[section][experiment_name]
            section_name = section
            break
    
    if experiment_config is None:
        available_experiments = []
        for section in ['experiments', 'quick_tests', 'production_configs', 'ablation_studies']:
            for exp_name in config_data.get(section, {}):
                available_experiments.append(f"{section}.{exp_name}")
        
        raise ValueError(f"Experiment '{experiment_name}' not found. Available: {available_experiments}")
    
    # Handle inheritance
    if 'inherit_from' in experiment_config:
        base_name = experiment_config['inherit_from']
        base_config = None
        
        # Find base config
        for section in ['experiments', 'quick_tests', 'production_configs', 'ablation_studies']:
            if base_name in config_data.get(section, {}):
                base_config = deepcopy(config_data[section][base_name])
                break
        
        if base_config is None:
            raise ValueError(f"Base experiment '{base_name}' not found")
        
        # Merge configs (experiment_config overrides base_config)
        merged_config = deepcopy(base_config)
        for key, value in experiment_config.items():
            if key == 'inherit_from':
                continue
            if isinstance(value, dict) and key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        experiment_config = merged_config
    
    return experiment_config, section_name


def config_to_args(config: dict, data_dir: str):
    """Convert configuration dictionary to argument list."""
    args = [data_dir]
    
    # Model parameters
    model = config.get('model', {})
    args.extend(['--latent-dim', str(model.get('latent_dim', 64))])
    args.extend(['--encoder-hidden-dim', str(model.get('encoder_hidden_dim', 256))])
    args.extend(['--cnf-hidden-dim', str(model.get('cnf_hidden_dim', 128))])
    args.extend(['--latent-cnf-hidden-dim', str(model.get('latent_cnf_hidden_dim', 128))])
    
    # Training parameters
    training = config.get('training', {})
    args.extend(['--epochs', str(training.get('epochs', 50))])
    args.extend(['--batch-size', str(training.get('batch_size', 8))])
    args.extend(['--lr', str(training.get('lr', 1e-3))])
    args.extend(['--beta-schedule', str(training.get('beta_schedule', 'linear'))])
    
    # Validation parameters
    validation = config.get('validation', {})
    args.extend(['--validation-frequency', str(validation.get('validation_frequency', 5))])
    args.extend(['--monitoring-frequency', str(validation.get('monitoring_frequency', 10))])
    args.extend(['--save-every', str(validation.get('save_frequency', 5))])
    
    # Data parameters
    data = config.get('data', {})
    if 'specific_slice' in data:
        args.extend(['--validation-slice', data['specific_slice']])
    
    # Add auto-resume by default
    args.append('--auto-resume')
    
    return args


def calculate_model_size(config: dict):
    """Calculate approximate model size in parameters."""
    model = config.get('model', {})
    
    latent_dim = model.get('latent_dim', 64)
    encoder_hidden = model.get('encoder_hidden_dim', 256)
    cnf_hidden = model.get('cnf_hidden_dim', 128)
    latent_cnf_hidden = model.get('latent_cnf_hidden_dim', 128)
    
    # Rough parameter estimation
    # Encoder: input processing + feature extraction + VAE heads
    encoder_params = (2 * encoder_hidden + encoder_hidden * encoder_hidden + 
                     encoder_hidden * latent_dim * 2)
    
    # Point CNF: context + ODE function layers
    point_cnf_params = (latent_dim + 2) * cnf_hidden + cnf_hidden * cnf_hidden * 3
    
    # Latent CNF: ODE function layers
    latent_cnf_params = latent_dim * latent_cnf_hidden + latent_cnf_hidden * latent_cnf_hidden * 3
    
    total_params = encoder_params + point_cnf_params + latent_cnf_params
    
    return {
        'total': total_params,
        'encoder': encoder_params,
        'point_cnf': point_cnf_params,
        'latent_cnf': latent_cnf_params
    }


def print_experiment_info(experiment_name: str, config: dict, section_name: str):
    """Print detailed experiment information."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name} ({section_name})")
    print(f"{'='*60}")
    
    print(f"\nDescription: {config.get('description', 'No description provided')}")
    
    # Model configuration
    print(f"\nModel Configuration:")
    model = config.get('model', {})
    for key, value in model.items():
        print(f"  {key}: {value}")
    
    # Estimated parameters
    param_estimate = calculate_model_size(config)
    print(f"\nEstimated Parameters:")
    print(f"  Total: ~{param_estimate['total']:,}")
    print(f"  Encoder: ~{param_estimate['encoder']:,}")
    print(f"  Point CNF: ~{param_estimate['point_cnf']:,}")
    print(f"  Latent CNF: ~{param_estimate['latent_cnf']:,}")
    
    # Training configuration
    print(f"\nTraining Configuration:")
    training = config.get('training', {})
    for key, value in training.items():
        print(f"  {key}: {value}")
    
    # Validation configuration
    print(f"\nValidation Configuration:")
    validation = config.get('validation', {})
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Data configuration
    print(f"\nData Configuration:")
    data = config.get('data', {})
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")


def list_experiments(config_path: str):
    """List all available experiments."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    print(f"\nAvailable Experiments in {config_path}:")
    print("="*50)
    
    for section in ['experiments', 'quick_tests', 'production_configs', 'ablation_studies']:
        if section in config_data:
            print(f"\n{section.upper()}:")
            for exp_name, exp_config in config_data[section].items():
                description = exp_config.get('description', 'No description')
                print(f"  {exp_name}: {description}")
    
    print("\nUsage: python scripts/run_experiment.py --config config/experiments.json --experiment <name> <data_dir>")


def main():
    parser = argparse.ArgumentParser(description="Run experiments using centralized configuration")
    parser.add_argument("data_dir", nargs='?', help="Directory containing .npy slice files")
    parser.add_argument("--config", type=str, default="config/experiments.json", 
                       help="Path to configuration file")
    parser.add_argument("--experiment", type=str, help="Experiment name to run")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--info", action="store_true", help="Show experiment info without running")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (overrides config)")
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        return
    
    if args.list:
        list_experiments(config_path)
        return
    
    if not args.experiment:
        print("ERROR: --experiment is required (or use --list to see available experiments)")
        list_experiments(config_path)
        return
    
    if not args.data_dir:
        print("ERROR: data_dir is required")
        return
    
    # Load experiment configuration
    try:
        experiment_config, section_name = load_config(config_path, args.experiment)
    except ValueError as e:
        print(f"ERROR: {e}")
        list_experiments(config_path)
        return
    
    # Show experiment info
    print_experiment_info(args.experiment, experiment_config, section_name)
    
    if args.info:
        return
    
    # Convert config to arguments
    train_args = config_to_args(experiment_config, args.data_dir)
    
    # Add seed override
    train_args.extend(['--seed', str(args.seed)])
    
    print(f"\nRunning training with arguments: {' '.join(train_args)}")
    print(f"Starting experiment: {args.experiment}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run training by calling train_with_validation's main function
    # Temporarily modify sys.argv
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['train_with_validation.py'] + train_args
        train_main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"ERROR during training: {e}")
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
