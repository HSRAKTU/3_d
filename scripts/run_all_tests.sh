#!/bin/bash

# Comprehensive CNF Test Suite Runner
# Run this to get maximum insights before building final architecture

echo "🧪 COMPREHENSIVE CNF BEHAVIOR TEST SUITE"
echo "========================================"
echo "This will run all tests to understand CNF behavior"
echo "Total tests: 8"
echo "Estimated time: 20-25 minutes on GPU"
echo

DATA_PATH="data/"
SLICE_NAME="single_slice_test.npy"
DEVICE="cuda"

# Create output directory
mkdir -p outputs/test_suite_results

# Test 1: Architecture Ablation
echo "📊 Test 1/6: CNF Architecture Ablation Study"
echo "Testing different hidden dimensions and solver steps..."
python scripts/test_cnf_ablation.py $DATA_PATH --slice-name $SLICE_NAME --device $DEVICE
echo "✓ Ablation study complete"
echo

# Test 2: Latent CNF
echo "📊 Test 2/6: Latent CNF Behavior"
echo "Testing the CNF that transforms latent distributions..."
python scripts/test_latent_cnf.py $DATA_PATH --slice-name $SLICE_NAME --device $DEVICE
echo "✓ Latent CNF test complete"
echo

# Test 3: Integration
echo "📊 Test 3/6: Encoder-Decoder Integration"
echo "Testing how encoder and decoder work together..."
python scripts/test_integration.py $DATA_PATH --slice-name $SLICE_NAME --device $DEVICE
echo "✓ Integration test complete"
echo

# Test 4: Scalability
echo "📊 Test 4/6: Point Count Scalability"
echo "Testing performance with different point counts..."
python scripts/test_point_count_scalability.py $DATA_PATH --slice-name $SLICE_NAME --device $DEVICE
echo "✓ Scalability test complete"
echo

# Test 5: Solver Comparison
echo "📊 Test 5/6: ODE Solver Comparison"
echo "Comparing Euler vs adaptive solvers..."
python scripts/test_solver_comparison.py $DATA_PATH --slice-name $SLICE_NAME --device $DEVICE
echo "✓ Solver comparison complete"
echo

# Test 6: VAE Configurations
echo "📊 Test 6/8: Complete VAE Configuration Test"
echo "Testing different VAE architectures..."
python scripts/test_vae_configurations.py $DATA_PATH --slice-name $SLICE_NAME --device $DEVICE
echo "✓ VAE configuration test complete"
echo

# Test 7: Memory Efficiency
echo "📊 Test 7/8: Memory and Batch Processing Test"
echo "Testing memory usage and batch capabilities..."
python scripts/test_memory_efficiency.py $DATA_PATH --slice-name $SLICE_NAME --device $DEVICE
echo "✓ Memory efficiency test complete"
echo

# Test 8: Slice Diversity
echo "📊 Test 8/8: Slice Diversity Test"
echo "Testing on different types of slices..."
python scripts/test_slice_diversity.py $DATA_PATH --device $DEVICE
echo "✓ Slice diversity test complete"
echo

# Generate summary report
echo "📝 Generating summary report..."
python scripts/generate_test_report.py
echo

echo "🎉 ALL TESTS COMPLETE!"
echo "Check the outputs/ directory for detailed results:"
echo "  - outputs/cnf_ablation/"
echo "  - outputs/latent_cnf_test/"
echo "  - outputs/integration_test/"
echo "  - outputs/scalability_test/"
echo "  - outputs/solver_comparison/"
echo "  - outputs/vae_configurations/"
echo "  - outputs/memory_efficiency/"
echo "  - outputs/slice_diversity/"
echo "  - outputs/test_suite_results/summary_report.md"
