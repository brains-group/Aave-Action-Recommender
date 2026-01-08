#!/bin/bash
# Quick script to start simulations with default settings (16 workers)
# Usage: ./run_simulations_quick.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_simulations_tmux.sh" 16 output7.log
