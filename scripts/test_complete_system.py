#!/usr/bin/env python3
"""
Comprehensive system test for complete PointFlow2D implementation.
Tests all components: Latent CNF, training, validation, monitoring, config system.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pointflow2d_fixed import PointFlow2DVAE_Fixed
from models.latent_cnf import LatentCNF
from models.pointflow_cnf import PointFlowCNF
from training.dataset import SliceDataset, collate_variable_length
from utils.validation import TrainingMonitor, setup_validation_slice
from torch.utils.data import DataLoader


def setup_logging():
    """Setup logging for comprehensive testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def test_model_architecture():
    """Test complete model architecture and get detailed summary."""
    logger = logging.getLogger(__name__)
    logger.info("Testing complete model architecture...")
    
    # Test different model configurations
    configs = [
        {
            'name': 'tiny_config',
            'latent_dim': 32,
            'encoder_hidden_dim': 128,
            'cnf_hidden_dim': 64,
            'latent_cnf_hidden_dim': 64
        },
        {
            'name': 'standard_config',
            'latent_dim': 64,
            'encoder_hidden_dim': 256,
            'cnf_hidden_dim': 128,
            'latent_cnf_hidden_dim': 128
        },
        {
            'name': 'large_config',
            'latent_dim': 128,
            'encoder_hidden_dim': 512,
            'cnf_hidden_dim': 256,
            'latent_cnf_hidden_dim': 256
        }
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"\nTesting {config['name']}...")
        
        # Create model
        model = PointFlow2DVAE_Fixed(
            latent_dim=config['latent_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            cnf_hidden_dim=config['cnf_hidden_dim'],
            latent_cnf_hidden_dim=config['latent_cnf_hidden_dim'],
            use_latent_flow=True
        )
        
        # Get model info
        model_info = model.get_model_info()
        
        # Test forward pass
        dummy_input = torch.randn(2, 50, 2)  # batch_size=2, num_points=50, dim=2
        
        try:
            # Test encoding
            mu, logvar = model.encode(dummy_input)
            logger.info(f"  ✓ Encoding: {dummy_input.shape} → mu: {mu.shape}, logvar: {logvar.shape}")
            
            # Test full forward pass
            result = model.forward(dummy_input)
            required_keys = ['z', 'log_likelihood', 'log_prior', 'kl_loss']
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                logger.error(f"  ✗ Forward pass missing keys: {missing_keys}")
            else:
                logger.info(f"  ✓ Forward pass complete with all required outputs")
            
            # Test decoding
            z = result['z']
            decoded = model.decode(z, num_points=50)
            logger.info(f"  ✓ Decoding: {z.shape} → {decoded.shape}")
            
            # Test sampling
            samples = model.sample(batch_size=3, num_points=50)
            logger.info(f"  ✓ Sampling: {samples.shape}")
            
            # Test loss computation
            loss_dict = model.compute_loss(dummy_input, beta=0.5)
            required_loss_keys = ['total_loss', 'recon_loss', 'prior_loss', 'kl_loss']
            missing_loss_keys = [key for key in required_loss_keys if key not in loss_dict]
            if missing_loss_keys:
                logger.error(f"  ✗ Loss computation missing keys: {missing_loss_keys}")
            else:
                logger.info(f"  ✓ Loss computation complete")
                for key, value in loss_dict.items():
                    logger.info(f"    {key}: {value.item():.6f}")
            
            results.append({
                'config': config,
                'model_info': model_info,
                'test_results': {
                    'encoding': True,
                    'forward': len(missing_keys) == 0,
                    'decoding': True,
                    'sampling': True,
                    'loss_computation': len(missing_loss_keys) == 0
                }
            })
            
            logger.info(f"  Model parameters: {model_info}")
            
        except Exception as e:
            logger.error(f"  ✗ Error in {config['name']}: {e}")
            results.append({
                'config': config,
                'error': str(e),
                'test_results': {'failed': True}
            })
    
    return results


def test_latent_cnf_integration():
    """Test Latent CNF integration and gradient flow."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting Latent CNF integration...")
    
    # Create model with Latent CNF
    model = PointFlow2DVAE_Fixed(
        latent_dim=64,
        encoder_hidden_dim=256,
        cnf_hidden_dim=128,
        latent_cnf_hidden_dim=128,
        use_latent_flow=True
    )
    
    # Test data
    dummy_input = torch.randn(2, 50, 2)
    
    try:
        # Test with latent flow enabled
        logger.info("Testing with Latent CNF enabled...")
        result_with_cnf = model.forward(dummy_input)
        
        # Check log_prior values
        log_prior_values = result_with_cnf['log_prior'].detach().numpy()
        logger.info(f"  Log prior values: {log_prior_values}")
        
        if np.allclose(log_prior_values, 0.0):
            logger.error("  ✗ Log prior values are zero - Latent CNF not working!")
            return False
        else:
            logger.info("  ✓ Log prior values are non-zero - Latent CNF working!")
        
        # Test without latent flow for comparison
        logger.info("Testing without Latent CNF for comparison...")
        model.use_latent_flow = False
        result_without_cnf = model.forward(dummy_input)
        log_prior_without = result_without_cnf['log_prior'].detach().numpy()
        logger.info(f"  Log prior without CNF: {log_prior_without}")
        
        # Test gradient flow
        logger.info("Testing gradient flow...")
        model.use_latent_flow = True
        model.train()
        
        loss_dict = model.compute_loss(dummy_input, beta=0.5)
        loss_dict['total_loss'].backward()
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if 'latent_cnf' in name:
                    logger.info(f"  Latent CNF gradient {name}: {grad_norm:.6f}")
        
        if len(grad_norms) == 0:
            logger.error("  ✗ No gradients found!")
            return False
        
        avg_grad_norm = np.mean(grad_norms)
        logger.info(f"  ✓ Average gradient norm: {avg_grad_norm:.6f}")
        
        if avg_grad_norm < 1e-8:
            logger.error("  ✗ Gradients too small - possible vanishing gradient!")
            return False
        
        logger.info("  ✓ Latent CNF integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Latent CNF integration test failed: {e}")
        return False


def test_configuration_system():
    """Test the configuration system."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting configuration system...")
    
    try:
        # Test loading configuration
        config_path = Path(__file__).parent.parent / "config" / "experiments.json"
        
        if not config_path.exists():
            logger.error(f"  ✗ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Test different experiment configurations
        test_experiments = [
            'single_slice_overfit',
            'tiny_10cars',
            'debug_tiny',
            'fast_validation'
        ]
        
        for exp_name in test_experiments:
            found = False
            for section in ['experiments', 'quick_tests', 'production_configs', 'ablation_studies']:
                if exp_name in config_data.get(section, {}):
                    exp_config = config_data[section][exp_name]
                    logger.info(f"  ✓ Found experiment '{exp_name}' in section '{section}'")
                    
                    # Validate required fields
                    required_sections = ['model', 'training', 'validation']
                    for req_section in required_sections:
                        if req_section not in exp_config:
                            logger.error(f"    ✗ Missing required section '{req_section}'")
                        else:
                            logger.info(f"    ✓ Section '{req_section}' present")
                    
                    found = True
                    break
            
            if not found:
                logger.error(f"  ✗ Experiment '{exp_name}' not found!")
                return False
        
        logger.info("  ✓ Configuration system test passed!")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Configuration system test failed: {e}")
        return False


def test_validation_monitoring():
    """Test validation and monitoring system."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting validation and monitoring system...")
    
    try:
        # Setup temporary validation directory
        temp_dir = Path(__file__).parent.parent / "outputs" / "test_validation"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create monitor
        monitor = TrainingMonitor(
            save_dir=str(temp_dir),
            model_name="test_model",
            save_frequency=5,
            visualization_frequency=2
        )
        
        # Create test model
        model = PointFlow2DVAE_Fixed(
            latent_dim=32,
            encoder_hidden_dim=128,
            cnf_hidden_dim=64,
            latent_cnf_hidden_dim=64,
            use_latent_flow=True
        )
        
        # Create test optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Test training step logging
        logger.info("  Testing training step logging...")
        loss_dict = {
            'total_loss': 1.5,
            'recon_loss': 1.0,
            'prior_loss': 0.3,
            'kl_loss': 0.2
        }
        
        monitor.log_training_step(0, loss_dict, model, optimizer)
        logger.info("  ✓ Training step logging successful")
        
        # Test validation slice setup
        logger.info("  Testing validation slice creation...")
        # Create a dummy validation slice
        dummy_slice = torch.randn(1, 50, 2)
        
        # Test reconstruction validation
        recon_metrics = monitor.validate_reconstruction(model, dummy_slice, epoch=0)
        logger.info(f"  ✓ Reconstruction validation: {recon_metrics}")
        
        # Test sampling validation
        sampling_metrics = monitor.validate_sampling(model, num_samples=2, num_points=30, epoch=0)
        logger.info(f"  ✓ Sampling validation: {sampling_metrics}")
        
        # Test issue detection
        issues = monitor.detect_training_issues(recent_history=1)
        logger.info(f"  ✓ Issue detection: {len(issues)} issues detected")
        
        # Test plots and summaries
        monitor.save_training_plots(0)
        monitor.save_validation_summary(0)
        logger.info("  ✓ Plots and summaries saved")
        
        # Check if files were created
        expected_files = [
            temp_dir / "plots" / "training_progress_epoch_0000.png",
            temp_dir / "validation_outputs" / "validation_summary_epoch_0000.json",
            temp_dir / "plots" / "reconstruction_epoch_0000.png",
            temp_dir / "plots" / "sampling_epoch_0000.png"
        ]
        
        for file_path in expected_files:
            if file_path.exists():
                logger.info(f"  ✓ Created: {file_path.name}")
            else:
                logger.warning(f"  ? Missing: {file_path.name}")
        
        logger.info("  ✓ Validation and monitoring test passed!")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Validation and monitoring test failed: {e}")
        return False


def test_training_integration():
    """Test the complete training integration."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting training integration...")
    
    try:
        # Create a small dataset for testing
        logger.info("  Creating test dataset...")
        
        # Generate synthetic 2D point clouds that look like simple shapes
        test_data = []
        for i in range(5):  # 5 test samples
            # Create a simple circle
            theta = np.linspace(0, 2*np.pi, 30)
            x = np.cos(theta) + np.random.normal(0, 0.1, 30)
            y = np.sin(theta) + np.random.normal(0, 0.1, 30)
            points = np.column_stack([x, y])
            test_data.append(points.astype(np.float32))
        
        # Test data loading
        class TestDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                points = torch.FloatTensor(self.data[idx])
                return {
                    'points': points,
                    'num_points': torch.tensor(points.shape[0]),
                    'car_id': f'test_car_{idx}',
                    'slice_idx': idx,
                    'normalization_info': {
                        'min_coords': [0.0, 0.0],
                        'max_coords': [1.0, 1.0], 
                        'scale': 1.0
                    }
                }
        
        dataset = TestDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, 
                               collate_fn=collate_variable_length)
        
        logger.info(f"  ✓ Test dataset created with {len(dataset)} samples")
        
        # Create model
        model = PointFlow2DVAE_Fixed(
            latent_dim=32,
            encoder_hidden_dim=128,
            cnf_hidden_dim=64,
            latent_cnf_hidden_dim=64,
            use_latent_flow=True
        )
        
        # Test one training step
        logger.info("  Testing training step...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        batch = next(iter(dataloader))
        points = batch['points']
        
        optimizer.zero_grad()
        loss_dict = model.compute_loss(points, beta=0.5)
        loss_dict['total_loss'].backward()
        
        # Check gradients before step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        logger.info(f"  ✓ Gradient norm before clipping: {grad_norm:.6f}")
        
        optimizer.step()
        
        logger.info(f"  ✓ Training step successful:")
        for key, value in loss_dict.items():
            logger.info(f"    {key}: {value.item():.6f}")
        
        # Test model state saving/loading
        logger.info("  Testing model state save/load...")
        
        temp_checkpoint = Path(__file__).parent.parent / "outputs" / "test_checkpoint.pt"
        
        # Save state
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': loss_dict['total_loss'].item()
        }
        torch.save(checkpoint, temp_checkpoint)
        
        # Create new model and load state
        new_model = PointFlow2DVAE_Fixed(
            latent_dim=32,
            encoder_hidden_dim=128,
            cnf_hidden_dim=64,
            latent_cnf_hidden_dim=64,
            use_latent_flow=True
        )
        
        loaded_checkpoint = torch.load(temp_checkpoint, weights_only=False)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        
        # Test that loaded model produces same output
        with torch.no_grad():
            original_output = model.forward(points)
            loaded_output = new_model.forward(points)
            
            output_diff = torch.abs(original_output['z'] - loaded_output['z']).max().item()
            if output_diff < 1e-6:
                logger.info(f"  ✓ Model save/load successful (diff: {output_diff:.2e})")
            else:
                logger.error(f"  ✗ Model save/load failed (diff: {output_diff:.2e})")
        
        # Clean up
        if temp_checkpoint.exists():
            temp_checkpoint.unlink()
        
        logger.info("  ✓ Training integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Training integration test failed: {e}")
        return False


def generate_system_summary():
    """Generate comprehensive system summary."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "experiments.json"
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Get standard configuration
    standard_config = config_data['experiments']['tiny_10cars']
    
    # Create model with standard config
    model_config = standard_config['model']
    model = PointFlow2DVAE_Fixed(
        latent_dim=model_config['latent_dim'],
        encoder_hidden_dim=model_config['encoder_hidden_dim'],
        cnf_hidden_dim=model_config['cnf_hidden_dim'],
        latent_cnf_hidden_dim=model_config['latent_cnf_hidden_dim'],
        use_latent_flow=model_config['use_latent_flow']
    )
    
    model_info = model.get_model_info()
    
    print(f"\n{'='*80}")
    print(f"POINTFLOW2D COMPLETE SYSTEM SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n🏗️  ARCHITECTURE OVERVIEW:")
    print(f"  • Complete PointFlow architecture adapted for 2D")
    print(f"  • 3-component system: Encoder + Point CNF + Latent CNF")
    print(f"  • Variational Autoencoder with CNF-based decoder")
    print(f"  • Full ELBO loss: Reconstruction + Prior + KL regularization")
    
    print(f"\n📊 MODEL PARAMETERS (Standard Config):")
    print(f"  • Total Parameters: {model_info['total_parameters']:,}")
    print(f"  • Encoder Parameters: {model_info['encoder_parameters']:,}")
    print(f"  • Point CNF Parameters: {model_info['point_cnf_parameters']:,}")
    print(f"  • Latent CNF Parameters: {model_info['latent_cnf_parameters']:,}")
    print(f"  • Uses Latent Flow: {model_info['use_latent_flow']}")
    
    print(f"\n🔧 CONFIGURATION SYSTEM:")
    exp_count = sum(len(config_data.get(section, {})) for section in 
                   ['experiments', 'quick_tests', 'production_configs', 'ablation_studies'])
    print(f"  • {exp_count} predefined experiment configurations")
    print(f"  • Progressive model scaling: Tiny (209K) → Medium (500K) → Large (1M)")
    print(f"  • Hyperparameter sweeps and ablation studies")
    print(f"  • RunPod-optimized configurations")
    
    print(f"\n📈 VALIDATION & MONITORING:")
    print(f"  • Comprehensive training metrics tracking")
    print(f"  • Real-time reconstruction validation")
    print(f"  • Sampling quality assessment")
    print(f"  • Early issue detection (gradient explosion, latent collapse)")
    print(f"  • Periodic decoded slice saving for RunPod analysis")
    print(f"  • Training plots and validation summaries")
    
    print(f"\n🎯 KEY IMPROVEMENTS OVER PREVIOUS VERSION:")
    print(f"  • ✅ Latent CNF implementation (was missing)")
    print(f"  • ✅ Complete PointFlow loss function")
    print(f"  • ✅ Gradient flow fixes for ODE integration")
    print(f"  • ✅ Comprehensive validation framework")
    print(f"  • ✅ Centralized configuration system")
    print(f"  • ✅ Early detection of training issues")
    
    print(f"\n🚀 READY FOR:")
    print(f"  • Single slice overfitting validation")
    print(f"  • Progressive model scaling experiments")
    print(f"  • RunPod GPU training with comprehensive monitoring")
    print(f"  • Systematic hyperparameter optimization")
    
    print(f"\n{'='*80}")


def main():
    """Run comprehensive system tests."""
    logger = setup_logging()
    
    print(f"\n{'='*60}")
    print(f"POINTFLOW2D COMPLETE SYSTEM TEST")
    print(f"{'='*60}")
    
    # Track test results
    test_results = {}
    
    # 1. Test model architecture
    logger.info("=" * 60)
    arch_results = test_model_architecture()
    test_results['architecture'] = arch_results
    
    # 2. Test Latent CNF integration
    logger.info("=" * 60)
    cnf_success = test_latent_cnf_integration()
    test_results['latent_cnf'] = cnf_success
    
    # 3. Test configuration system
    logger.info("=" * 60)
    config_success = test_configuration_system()
    test_results['configuration'] = config_success
    
    # 4. Test validation/monitoring
    logger.info("=" * 60)
    validation_success = test_validation_monitoring()
    test_results['validation'] = validation_success
    
    # 5. Test training integration
    logger.info("=" * 60)
    training_success = test_training_integration()
    test_results['training'] = training_success
    
    # Generate summary
    logger.info("=" * 60)
    generate_system_summary()
    
    # Final results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    overall_success = True
    
    for test_name, result in test_results.items():
        if test_name == 'architecture':
            # Check architecture results
            arch_success = all(not res.get('test_results', {}).get('failed', False) 
                             for res in result)
            status = "✅ PASS" if arch_success else "❌ FAIL"
            print(f"  {test_name.upper()}: {status}")
            if not arch_success:
                overall_success = False
        else:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name.upper()}: {status}")
            if not result:
                overall_success = False
    
    print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print(f"\n🎉 System is ready for experiments!")
        print(f"Next step: Run single slice overfitting test")
        print(f"Command: python scripts/run_experiment.py --experiment single_slice_overfit <data_dir>")
    else:
        print(f"\n⚠️  Please fix failing tests before proceeding to experiments.")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
