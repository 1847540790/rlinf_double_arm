#!/bin/bash
# Run PPO Training (with or without HIL)
#
# Usage:
#   ./run_ppo.sh                  # Standard PPO (no HIL)
#   ./run_ppo.sh --hil            # PPO with HIL (keyboard control)
#   ./run_ppo.sh --hil --mock     # PPO with HIL (mock device, for testing)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$SCRIPT_DIR"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH}"

# Default settings
CONFIG="config/ppo_standard.yaml"
MAX_STEPS=10000
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hil)
            CONFIG="config/ppo_hil.yaml"
            EXTRA_ARGS="$EXTRA_ARGS --enable_hil"
            shift
            ;;
        --no-hil)
            CONFIG="config/ppo_standard.yaml"
            EXTRA_ARGS="$EXTRA_ARGS --no_hil"
            shift
            ;;
        --mock)
            # Use mock input device (for testing without keyboard capture)
            echo "Note: Using mock input device for testing"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --resume)
            EXTRA_ARGS="$EXTRA_ARGS --resume $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "PPO Training"
echo "============================================"
echo "Config: $CONFIG"
echo "Max steps: $MAX_STEPS"
echo "============================================"

python train_ppo_hil.py \
    --config "$CONFIG" \
    --max_steps "$MAX_STEPS" \
    $EXTRA_ARGS

echo "Done!"

