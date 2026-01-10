#!/bin/bash
# Script to run performSimulations.py in a tmux session with specified number of workers
# Usage: ./run_simulations_tmux.sh [num_workers] [log_file]

set -e

# Default values
NUM_WORKERS=${1:-16}
LOG_FILE=${2:-"output7.log"}
SESSION_NAME="simulations"
CONDA_ENV=${CONDA_ENV:-"aave-action-recommender"}

# Get script directory and project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory (where performSimulations.py is located)
cd "$PROJECT_ROOT"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Warning: tmux session '$SESSION_NAME' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session and create new: tmux kill-session -t $SESSION_NAME && $0 $@"
    exit 1
fi

# Create new tmux session (detached)
echo "Creating tmux session '$SESSION_NAME' with $NUM_WORKERS workers..."
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT"

# Send command to run simulations (from project root)
# Activate conda environment and run the script
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_ROOT && source ~/.bashrc && conda activate $CONDA_ENV && python performSimulations.py --workers $NUM_WORKERS --log-file $LOG_FILE" C-m

echo ""
echo "✓ Simulations started in tmux session '$SESSION_NAME'"
echo ""
echo "Useful commands:"
echo "  - Attach to session:    tmux attach -t $SESSION_NAME"
echo "  - Detach from session:  Press Ctrl+B, then D"
echo "  - View logs:            tail -f $LOG_FILE"
echo "  - Check progress:       ./scripts/check_progress.sh"
echo "  - Kill session:         tmux kill-session -t $SESSION_NAME"
echo ""
echo "To check progress, run: ./check_progress.sh"
