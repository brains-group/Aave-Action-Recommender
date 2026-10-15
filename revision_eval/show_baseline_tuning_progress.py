"""One read-only dashboard for the chronological baseline tuning study."""
import argparse
import json
import time
from pathlib import Path

from .run_baseline_tuning import DEFAULT_RUN, ROOT, job_id


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    if seconds is None:
        return 'waiting for timings'
    n = max(0, int(seconds))
    return f'{n//3600}h {(n%3600)//60:02d}m {n%60:02d}s'


def show(run):
    manifest = read(run / 'manifest.json')
    progress = read(run / 'progress.json')
    phase = progress.get('phase', 'preparing')
    heartbeat = progress.get('heartbeat', 0)
    terminal = phase in ('complete', 'needs_attention')
    freshness = 'saved final status' if terminal else ('live heartbeat' if time.time() - heartbeat < 30 else 'STALE: check runner.log')
    corrections = read(run / 'correction-jobs.json')
    cases = read(run / 'calibration-cases.json') or []
    corrected = sum((run / 'results' / (job_id(key) + '.pkl')).exists() for key in corrections)
    finished_cal = sum((run / 'calibration' / c['case_id'] / 'complete.json').exists() for c in cases)
    outcomes = sum(len(list((run / 'calibration' / c['case_id']).glob('*-*.pkl'))) for c in cases)
    timings = [read(p) for p in (run / 'simulation-timings').glob('*.json')]
    active_roots = [run / 'active']
    if manifest.get('reuse_run'):
        active_roots.append(Path(manifest['reuse_run']) / 'active')
    active = [read(p) for root in active_roots for p in root.glob('*.json')]
    active = [a for a in active if time.time() - a.get('heartbeat', 0) < 20]
    frozen = read(run / 'frozen-setting.json')
    print(time.strftime('%Y-%m-%d %H:%M:%S'), phase, '(' + freshness + ')')
    print(f'Calibration: {finished_cal}/{len(cases)} checkpoints; {outcomes}/{len(cases)*8} configuration outcomes; all eight settings')
    print(f'Baseline corrections: {corrected}/{len(corrections)} distinct action inputs complete (cached pilot inputs included)')
    correction_summary_root = Path(manifest.get('reuse_run') or run)
    correction_summary = read(correction_summary_root / 'correction-summary-progress.json')
    if correction_summary.get('phase') == 'complete':
        print('Corrected baseline summary:', correction_summary_root / 'corrected-baseline-results.json')
    elif correction_summary.get('phase') == 'failed':
        print('Correction summary needs attention:', correction_summary.get('error'))
    if frozen:
        print('Frozen winning trigger/target:', frozen['configuration'])
    else:
        print('Winning-setting testing: waiting for calibration selection')
    print(f'New simulations saved: {len(timings)}; cumulative simulation time {duration(sum(t.get("seconds", 0) for t in timings))}')
    stage_eta = progress.get('task_rate_eta_seconds')
    eta_method = 'task rate; provisional'
    if phase == 'expanded_calibration' and finished_cal >= 4:
        calibration_work = sum(t.get('seconds', 0) for t in timings if t.get('stage') == 'calibration')
        # Use actual simulation work across completed cases, not near-zero
        # cache-hit task durations at the beginning of an expanded/resumed run.
        stage_eta = ((len(cases) - finished_cal) * calibration_work / finished_cal
                     / manifest.get('calibration_workers', 4) * 1.25)
        eta_method = 'measured work per case plus overhead; provisional'
    print(f'Current-stage tasks: {progress.get("completed_stage_tasks", "-")}/{progress.get("total_stage_tasks", "-")}; approximate stage ETA: {duration(stage_eta)} ({eta_method})')
    print('Failed tasks:', progress.get('failures', 0))
    if progress.get('error'):
        print('Needs attention:', progress['error'])
    for worker in sorted(active, key=lambda a: a.get('worker', 0)):
        elapsed = duration(time.time() - worker['started']) if worker.get('started') else '-'
        print(f'  Worker {worker["worker"]}: {worker["phase"]} {worker.get("case_id") or ""}; active {elapsed}; new {worker.get("new_simulations", 0)}, cache hits {worker.get("cache_hits", 0)}')
    print('Log:', run / 'runner.log')
    if phase == 'complete':
        print('Results:', run / 'matched-results.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--watch', type=float)
    args = parser.parse_args()
    if args.run is None:
        active = read(ROOT / 'outputs/baseline-tuning-active.json')
        args.run = Path(active['run']) if active.get('run') else DEFAULT_RUN
    if args.watch is not None and args.watch <= 0:
        parser.error('--watch must be positive')
    while True:
        if args.watch:
            print('\033[2J\033[H', end='')
        show(args.run.resolve())
        if args.watch is None:
            break
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
