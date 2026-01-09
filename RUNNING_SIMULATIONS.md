# Running Simulations in Background (tmux)

This guide explains how to run `performSimulations.py` in a background tmux session and monitor progress.

## Quick Start

### Option 1: Quick Start (Default 16 workers)
```bash
./scripts/run_simulations_quick.sh
```

### Option 2: Custom Workers
```bash
./scripts/run_simulations_tmux.sh 16
```

### Option 3: Custom Workers and Log File
```bash
./scripts/run_simulations_tmux.sh 16 output7.log
```

## Detailed Usage

### Start Simulations in tmux

```bash
./scripts/run_simulations_tmux.sh [num_workers] [log_file]
```

**Arguments:**
- `num_workers`: Number of parallel worker processes (default: 16)
- `log_file`: Custom log file name (default: output7.log)

**Examples:**
```bash
# Use 16 workers with default log file
./scripts/run_simulations_tmux.sh 16

# Use 8 workers with custom log file
./scripts/run_simulations_tmux.sh 8 custom_run.log

# Use all available cores (auto-detect)
./scripts/run_simulations_tmux.sh $(nproc)
```

### Check Progress

```bash
./scripts/check_progress.sh [log_file]
```

This script shows:
- Last 20 lines of the log
- Progress count (processed/total)
- Processing rate and ETA
- Error count and recent errors
- Whether the process is still running

**Examples:**
```bash
# Check default log file
./scripts/check_progress.sh

# Check custom log file
./scripts/check_progress.sh custom_run.log

# Detailed check with JSON statistics
./scripts/check_progress_detailed.sh
```

### tmux Commands

#### Attach to Session
```bash
tmux attach -t simulations
```

#### Detach from Session
Press `Ctrl+B`, then `D`

#### View Session Without Attaching
```bash
tmux capture-pane -t simulations -p
```

#### Kill Session
```bash
tmux kill-session -t simulations
```

#### List All Sessions
```bash
tmux ls
```

## Monitoring Options

### 1. Watch Progress (Updates Every 5 Seconds)
```bash
watch -n 5 ./scripts/check_progress.sh
```

### 2. Tail Log File in Real-Time
```bash
tail -f output7.log
```

### 3. Monitor Log with Timestamps
```bash
tail -f output7.log | ts '[%Y-%m-%d %H:%M:%S]'
```

### 4. Check Statistics JSON File
The script automatically saves statistics to:
```
cache/statistics/simulation_statistics_YYYYMMDD_HHMMSS.json
```

View the latest statistics:
```bash
ls -t cache/statistics/*.json | head -1 | xargs cat | jq .
```

### 5. Monitor System Resources
```bash
# In another terminal
htop
# or
watch -n 2 'ps aux | grep performSimulations'
```

## Running Directly (Without tmux)

You can also run directly with command-line arguments:

```bash
# Use 16 workers
python3 performSimulations.py --workers 16

# Use custom log file
python3 performSimulations.py --workers 16 --log-file my_log.log

# Auto-detect workers (uses all available cores)
python3 performSimulations.py
```

## Troubleshooting

### Session Already Exists
If you get "tmux session already exists":
```bash
# Attach to existing session
tmux attach -t simulations

# OR kill and restart
tmux kill-session -t simulations
./scripts/run_simulations_tmux.sh 16
```

### Process Not Running
Check if the process crashed:
```bash
# Check log for errors
tail -100 output7.log | grep -i error

# Check if tmux session exists
tmux ls

# Re-attach to see what happened
tmux attach -t simulations
```

### Out of Memory
If you encounter memory issues:
- Reduce number of workers: `./run_simulations_tmux.sh 8`
- Check memory usage: `free -h`
- Monitor processes: `ps aux --sort=-%mem | head -10`

### Check Disk Space
Simulations create cache files:
```bash
# Check cache directory size
du -sh cache/

# Check available disk space
df -h
```

## Expected Output Locations

1. **Log File**: `output7.log` (or custom name)
2. **Statistics JSON**: `cache/statistics/simulation_statistics_*.json`
3. **Cache Files**: `cache/simulation_results/*.pkl`

## Performance Tips

1. **Optimal Workers**: Use 16 workers for the idea-node-07 machine
2. **Monitor Progress**: Check every 10-15 minutes initially, then less frequently
3. **Disk Space**: Ensure sufficient space for cache files
4. **Network**: If using network storage, consider local cache first

## Stopping Simulations

### Graceful Stop
```bash
# Attach to session
tmux attach -t simulations

# Press Ctrl+C to stop gracefully
# The script will save statistics before exiting
```

### Force Kill
```bash
# Kill tmux session
tmux kill-session -t simulations

# OR kill process directly
pkill -f performSimulations.py
```

Note: Graceful stop is recommended as it saves final statistics.
