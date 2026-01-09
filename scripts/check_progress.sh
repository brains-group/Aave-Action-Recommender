#!/bin/bash
# Script to check progress of running simulations
# Usage: ./check_progress.sh [log_file]

LOG_FILE=${1:-"output7.log"}

# Get script directory and project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_PATH="$PROJECT_ROOT/$LOG_FILE"

# Check if log file exists
if [ ! -f "$LOG_PATH" ]; then
    echo "Error: Log file not found: $LOG_PATH"
    exit 1
fi

echo "=========================================="
echo "Simulation Progress Check"
echo "=========================================="
echo "Log file: $LOG_PATH"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Get last few lines of log
echo "--- Last 20 lines of log ---"
tail -n 20 "$LOG_PATH"
echo ""

# Count processed recommendations (with proper error handling)
# Look for the specific "Processed X/Y recommendations" pattern
# Be very specific to avoid matching dates or other numbers in the log line
PROCESSED_RAW=$(grep -E "Processed [0-9]+/[0-9]+ recommendations" "$LOG_PATH" 2>/dev/null | tail -1)

# If not found (case-insensitive), try again
if [ -z "$PROCESSED_RAW" ]; then
    PROCESSED_RAW=$(grep -iE "processed [0-9]+/[0-9]+ recommendations" "$LOG_PATH" 2>/dev/null | tail -1)
fi

if [ -n "$PROCESSED_RAW" ]; then
    # Extract the X/Y pattern specifically - must be between "Processed " and " recommendations"
    # Use sed to extract the pattern: "Processed X/Y recommendations" -> "X/Y"
    PATTERN=$(echo "$PROCESSED_RAW" | sed -nE 's/.*Processed[[:space:]]+([0-9]+\/[0-9]+)[[:space:]]+recommendations.*/\1/p' | head -1)
    
    if [ -n "$PATTERN" ] && [[ "$PATTERN" =~ ^[0-9]+/[0-9]+$ ]]; then
        # Extract numbers from the X/Y pattern
        PROCESSED=$(echo "$PATTERN" | cut -d'/' -f1)
        TOTAL=$(echo "$PATTERN" | cut -d'/' -f2)
    else
        # If pattern not found or invalid, set to zero (safer than guessing)
        PROCESSED="0"
        TOTAL="0"
    fi
else
    PROCESSED="0"
    TOTAL="0"
fi

# Ensure we have valid numbers (handle empty strings)
PROCESSED=${PROCESSED:-0}
TOTAL=${TOTAL:-0}

# Validate numbers are integers before arithmetic (suppress error output)
if [ -n "$PROCESSED" ] && [ -n "$TOTAL" ] && [ "$PROCESSED" -eq "$PROCESSED" ] 2>/dev/null && [ "$TOTAL" -eq "$TOTAL" ] 2>/dev/null; then
    if [ "$TOTAL" -gt 0 ] 2>/dev/null; then
        PERCENTAGE=$((PROCESSED * 100 / TOTAL))
        echo "Progress: $PROCESSED / $TOTAL recommendations ($PERCENTAGE%)"
        REMAINING=$((TOTAL - PROCESSED))
        if [ "$REMAINING" -ge 0 ] 2>/dev/null; then
            echo "Remaining: $REMAINING recommendations"
        fi
    else
        echo "Progress: Unable to determine (log may still be initializing or no progress data found)"
        echo "  Tip: Progress will appear after simulations start processing recommendations"
    fi
else
    echo "Progress: Unable to parse progress data from log"
    echo "  The log file may be from a previous run or simulation hasn't started yet"
fi

echo ""

# Check for errors
ERROR_COUNT=$(grep -i "error\|exception\|failed" "$LOG_PATH" | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Found $ERROR_COUNT error/exception messages in log"
    echo "   Last few errors:"
    grep -i "error\|exception\|failed" "$LOG_PATH" | tail -3 | sed 's/^/   /'
else
    echo "✓ No errors found in log"
fi

echo ""

# Get processing rate if available
RATE_LINE=$(grep -o "[0-9.]* recs/sec" "$LOG_PATH" | tail -1)
if [ ! -z "$RATE_LINE" ]; then
    echo "Processing rate: $RATE_LINE"
fi

# Get ETA if available
ETA_LINE=$(grep -o "ETA: [^)]*" "$LOG_PATH" | tail -1)
if [ ! -z "$ETA_LINE" ]; then
    echo "$ETA_LINE"
fi

echo ""

# Check if process is still running
if pgrep -f "performSimulations.py" > /dev/null; then
    echo "✓ Simulation process is running"
else
    echo "⚠️  Simulation process not found (may have completed or crashed)"
fi

echo ""
echo "=========================================="
echo "To view full log: tail -f $LOG_PATH"
echo "To attach to tmux: tmux attach -t simulations"
