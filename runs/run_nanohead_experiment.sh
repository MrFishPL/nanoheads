#!/bin/bash

# Nanohead Experiment Runner
# This script runs the nanohead experiment with proper environment setup
#
# Usage:
#   bash runs/run_nanohead_experiment.sh [OPTIONS]
#
# Options:
#   --target-params N           Target model size in millions (default: 50)
#   --nanohead-dim N            Nanohead dimension (default: 3)
#   --proportions "..."         Comma-separated proportions (default: "0.0,0.2,0.4,0.6,0.8")
#   --target-param-data-ratio N Data:param ratio for training duration (default: 10.5, Chinchilla=20)
#   --device-batch-size N       Per-device batch size (default: 32, reduce if OOM)
#
# Examples:
#   bash runs/run_nanohead_experiment.sh --target-params 100
#   bash runs/run_nanohead_experiment.sh --target-params 500
#   bash runs/run_nanohead_experiment.sh --proportions "0.0,0.3,0.5"
#   bash runs/run_nanohead_experiment.sh --target-params 500 --target-param-data-ratio 20
#   bash runs/run_nanohead_experiment.sh --target-params 500 --device-batch-size 16

set -e  # Exit on error

echo "=========================================="
echo "Nanohead Experiment"
echo "=========================================="
echo ""

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No .venv found. Make sure dependencies are installed."
fi

# Check if wandb is configured
if [ -z "$WANDB_API_KEY" ]; then
    echo ""
    echo "WARNING: WANDB_API_KEY not set!"
    echo "To enable wandb logging, either:"
    echo "  1. Run: wandb login"
    echo "  2. Set WANDB_API_KEY environment variable"
    echo ""
    echo "Proceeding anyway (wandb may use offline mode)..."
    echo ""
fi

# Set environment variables
export OMP_NUM_THREADS=1
export WANDB_PROJECT=nanochat

echo "Running nanohead experiment..."
echo "This will train models with different nanohead proportions."
echo "Expected runtime: ~5-10 minutes per model (depending on hardware)"
echo ""

# Run the experiment, passing all arguments through
python runs/nanohead_experiment.py "$@"

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
echo ""
echo "Results saved to: nanohead_experiment_results.png"
echo "Check your wandb dashboard for detailed logs."
