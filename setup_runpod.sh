#!/bin/bash
# RunPod Setup Script for PointFlow2D Single Slice Overfitting
# This script sets up the environment on RunPod with RTX 4090

echo "🚀 Setting up PointFlow2D on RunPod..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Make sure you're in the generative_3d_research directory."
    exit 1
fi

# Create logs directory [[memory:6220963]]
mkdir -p logs

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "📊 To run single slice overfitting test:"
echo "   python scripts/test_single_slice_overfit.py data/ \\"
echo "     --slice-name single_slice_test.npy \\"
echo "     --epochs 1000 \\"
echo "     --device cuda \\"
echo "     --viz-freq 50 \\"
echo "     --save-freq 10"
echo ""
echo "💡 Or for a full dataset test:"
echo "   python scripts/test_single_slice_overfit.py data/processed/single_car_slices \\"
echo "     --epochs 1000 \\"
echo "     --device cuda"
echo ""
echo "📈 Monitor progress in: outputs/single_slice_overfit/[timestamp]/"
