#!/bin/bash
# Detailed progress checker with JSON statistics summary
# Usage: ./check_progress_detailed.sh [log_file]

LOG_FILE=${1:-"output7.log"}

# Get script directory and project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_PATH="$PROJECT_ROOT/$LOG_FILE"

echo "=========================================="
echo "Detailed Simulation Progress"
echo "=========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run basic progress check
"$SCRIPT_DIR/check_progress.sh" "$LOG_FILE"

echo ""
echo "--- Statistics Summary ---"

# Find latest statistics JSON file
STATS_DIR="$PROJECT_ROOT/cache/statistics"
if [ -d "$STATS_DIR" ]; then
    LATEST_STATS=$(ls -t "$STATS_DIR"/*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_STATS" ] && [ -f "$LATEST_STATS" ]; then
        echo "Latest stats file: $(basename $LATEST_STATS)"
        echo ""
        
        # Extract key statistics if jq is available
        if command -v jq &> /dev/null; then
            echo "Overall Statistics:"
            jq -r '.overall | "  Processed: \(.processed // 0)\n  Liquidated without: \(.liquidated_without // 0)\n  Liquidated with: \(.liquidated_with // 0)\n  Improved: \(.improved // 0)\n  Worsened: \(.worsened // 0)\n  No change: \(.no_change // 0)"' "$LATEST_STATS" 2>/dev/null || echo "  (Unable to parse JSON)"
        else
            echo "  Install 'jq' for JSON parsing: sudo apt-get install jq"
            echo "  Raw stats file: $LATEST_STATS"
        fi
    else
        echo "No statistics JSON file found yet (simulation may still be initializing)"
    fi
else
    echo "Statistics directory not found"
fi

echo ""
echo "=========================================="
