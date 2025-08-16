#!/bin/bash
# Launch script for PointFlow2D VAE training on POD

# Default configuration
EXPERIMENT_NAME="vae_2d_experiment"
BATCH_SIZE=8
EPOCHS=500
LEARNING_RATE=5e-4
NUM_CARS=10
SOLVER_STEPS=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_cars)
            NUM_CARS="$2"
            shift 2
            ;;
        --stochastic)
            USE_STOCHASTIC="--use_stochastic"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
OUTPUT_DIR="outputs/vae_training/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Log configuration
echo "="*50
echo "PointFlow2D VAE Training"
echo "="*50
echo "Configuration:"
echo "  Name: $EXPERIMENT_NAME"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Num Cars: $NUM_CARS"
echo "  Solver Steps: $SOLVER_STEPS"
echo "  Stochastic: ${USE_STOCHASTIC:-False}"
echo "  Output: $OUTPUT_DIR"
echo "="*50

# Run training
python scripts/train_vae_2d.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --num_cars $NUM_CARS \
    --solver_steps $SOLVER_STEPS \
    --output_dir $OUTPUT_DIR \
    --lambda_recon 1.0 \
    --lambda_prior 0.1 \
    --lambda_entropy 0.01 \
    --lambda_chamfer 10.0 \
    --lambda_volume 0.01 \
    ${USE_STOCHASTIC} \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "Training complete! Results in: $OUTPUT_DIR"
