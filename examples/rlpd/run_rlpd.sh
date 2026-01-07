#!/bin/bash
# Run RLPD Training Script
# 
# Usage:
#   ./run_rlpd.sh              # Run with mock environment (for testing)
#   ./run_rlpd.sh --real       # Run with real robot config
#   ./run_rlpd.sh --resume checkpoint.pt  # Resume training

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to script directory
cd "$SCRIPT_DIR"

# Add rlinf to PYTHONPATH
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH}"

# Default config
CONFIG="config/rlpd_mock.yaml"
MAX_STEPS=10000
RESUME=""
DEMO_PATHS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --real)
            CONFIG="config/rlpd_real_robot.yaml"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --demo_paths)
            shift
            DEMO_PATHS="--demo_paths"
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DEMO_PATHS="$DEMO_PATHS $1"
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "RLPD Training (HIL-SERL style)"
echo "============================================"
echo "Config: $CONFIG"
echo "Max steps: $MAX_STEPS"
echo "PYTHONPATH: $PYTHONPATH"
echo "============================================"

# Run training
python train_rlpd.py \
    --config "$CONFIG" \
    --max_steps "$MAX_STEPS" \
    $RESUME \
    $DEMO_PATHS

echo "Training finished!"

