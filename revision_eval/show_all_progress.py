"""One read-only dashboard for reviewer experiments, including queued work.

Run from any directory: python -m revision_eval.show_all_progress --watch 10
The live runner emits a heartbeat; saved output is never proof of a live process.
"""
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600}h {(seconds%3600)//60:02d}m {seconds%60:02d}s'

def phase_state(stage, runner, finished=False):
    if finished:
        return 'DONE'
    if runner.get('stage') == stage:
        if runner.get('state') == 'running':
            heartbeat = runner.get('heartbeat_epoch')
            return 'RUNNING' if heartbeat and time.time()-heartbeat < 45 else 'STALE / CHECK LOG'
        return str(runner.get('state', 'unknown')).upper()
    return 'QUEUED' if runner.get('state') == 'running' else 'NOT RUNNING'

def rate_eta(progress):
    done = progress.get('completed_this_run', 0)
    seconds = progress.get('seconds', 0)
    remaining = progress.get('remaining_this_run', 0)
    return duration(seconds*remaining/done) if done >= 4 and seconds > 0 else 'waiting for rate'

def collect(root):
    run = root/'outputs/high-risk-shortest-20260925'
    audit = root/'outputs/revision-audits-20260928'
    runner = read(run/'status.json')
    study = run/'studies'/runner.get('study_key', 'fixed-2592000-seconds-trained-v1')
    cohort = read(run/'cohort/manifest.json')
    count = cohort.get('selected', 1200)
    generation = read(study/'generation-core/complete.json')
    evaluation = study/'agent-evaluation'
    baseline = study/'policy-baselines'
    progress = read(evaluation/'progress.json')
    outcomes = [read(p) for p in (evaluation/'case-status').glob('*.json')]
    saved = len(list((evaluation/'cases').glob('*.json')))
    active = [read(p) for p in (evaluation/'active').glob('*.json')]
    baseline_active = read(baseline/'active.json')
    rows = []
    def add(task, state, detail):
        rows.append(dict(task=task, state=state, detail=detail))
    add('Stress recommendations', 'DONE' if generation.get('cases') == count else 'INCOMPLETE',
        f"{generation.get('cases', 0)}/{count}; errors {generation.get('errors', '?')}")
    add('Stress agent evaluation', phase_state('agent_evaluation', runner, (evaluation/'complete.json').exists()),
        f'{saved}/{count*2} case-variants saved; {sum(bool(x.get("success")) for x in outcomes)} successful; '
        f'{sum(bool(x.get("paired")) for x in outcomes)} paired; '
        f'{sum(not x.get("success", False) for x in outcomes)} skipped/failed; '
        f'pass ETA {rate_eta(progress) if saved < count*2 else "finished"}')
    complete = len(list((baseline/'cases').glob('*/complete.json')))
    errors = len(list((baseline/'cases').glob('*/error.json')))
    unavailable = len(list((baseline/'cases').glob('*/unavailable.json')))
    add('Stress policy baselines', phase_state('policy_baselines', runner, (baseline/'pass-complete.json').exists()),
        f'{complete}/{count*2} case-variant folders complete; errors {errors}; unavailable {unavailable}; '
        '5 policies including control per available folder')
    if (baseline/'pass-complete.json').exists() and (errors or unavailable):
        rows[-1]['state'] = 'PASS DONE / EXCLUSIONS'
    for label, folder, detail in [
        ('Trend grid', 'trend-original-full', '12,000 original checkpoint rows; refreshed predictions'),
        ('Predictive uncertainty', 'predictive-uncertainty', '4 tasks; 300 account-cluster bootstrap draws'),
        ('Core policy comparison', 'policy-core-common', '11,848 common cases; historical funding'),
        ('Indexed policy comparison', 'policy-indexed-common', '5,471 common cases; historical funding'),
        ('Core market regimes', 'policy-core-regimes', 'trailing volatility and debt-size slices'),
        ('Indexed market regimes', 'policy-indexed-regimes', 'trailing volatility and debt-size slices'),
        ('Cohort lineage audit', 'cohort-lineage', 'complete audit; original 8,400-ID lineage unresolved'),
        ('Core wallet provenance', 'wallet-all-core', '11,500 attempted cases'),
        ('Indexed wallet provenance', 'wallet-all-indexed', '5,285 attempted cases'),
        ('Core funding exposure', 'wallet-exposure-core-v2', '448/692 rescues exposed in proposed asset; not corrected outcomes'),
        ('Indexed funding exposure', 'wallet-exposure-indexed-v2', '196/382 rescues exposed in proposed asset; not corrected outcomes'),
        ('Sensitivity diagnostics', 'sensitivity-fixed30d-v1', '8 cases × 7 scenarios; exclusions and warning metadata issue'),
        ('Prefix-funding replay', 'sensitivity-prefix-wallet-v1', '8 cases; 7 controls changed'),
        ('Indexed identity audit', 'indexed-collision-audit-v5', '1,556/1,556 classified; 560 checkpoint overlaps, 570 earlier, 426 neither'),
        ('Historical negative-case audit', 'historical-validation-v1', 'censor-aware seven-day metrics; cached historical source, not independent ground truth'),
    ]:
        add(label, 'DONE (diagnostic)' if (audit/folder/'summary.json').exists() else 'NOT COMPLETE', detail)
    for label, state, detail in [
        ('Affected-result reruns', 'NOT SCHEDULED', 'per author request; legacy outputs are not corrected validation'),
        ('Corrected funding / equal-capital study', 'DEFERRED', 'requires consistent historical state/funding; not currently running'),
        ('Indexed unresolved controls', 'DEFERRED', 'distinct-underlying profile/price identity; separate post-history snapshot pilot below'),
        ('Independent historical state validation', 'DEFERRED', 'no independent historical account-state references; source-consistency audit completed'),
        ('Final paper tables, figures and builds', 'PARTIAL', 'matched policy/stress table added and both PDFs built; remaining analyses and response letter pending'),
    ]:
        add(label, state, detail)
    funding = root/'outputs/revision-audits-20260929/funding-prefix-32-v1'
    if (funding/'manifest.json').exists():
        tracked = read(funding.parent/'funding-prefix-32-driver/status.json')
        p = read(funding/'progress.json')
        manifest = read(funding/'manifest.json')
        total = len(manifest.get('indices', []))*len(manifest.get('scenarios', []))
        state = str(tracked.get('state', 'not started')).upper()
        if state == 'RUNNING' and time.time()-tracked.get('heartbeat_epoch', 0)>45:
            state = 'STALE / CHECK LOG'
        funding_active = read(funding/'active.json')
        add('Additional prefix funding sensitivity', state,
            f"{p.get('saved',0)}/{total} case-scenarios; factors 1/1.5/2; "
            f"active {funding_active.get('index','-')}:{funding_active.get('scenario','-')}; "
            f"ETA {rate_eta(p) if state=='RUNNING' else 'finished' if state=='DONE' else 'unknown'}")
    warning = audit/'warning-metadata-v2'
    warning_rows = [read(path) for path in (warning/'cases').glob('*/*/result.json')]
    add('Warning-policy verification', 'DONE' if (warning/'summary.json').exists() else 'INCOMPLETE',
        f'{len(warning_rows)}/32 scenario cases; isolated corrected output labels; '
        f'{sum(not row.get("result_success",False) for row in warning_rows)} failed/skipped')
    mismatch_status = read(audit/'checkpoint-mismatch-driver/status.json')
    mismatch_state = str(mismatch_status.get('state','not started')).upper()
    if mismatch_state == 'RUNNING' and time.time()-mismatch_status.get('heartbeat_epoch',0)>45:
        mismatch_state = 'STALE / CHECK LOG'
    mismatch_rows = list((audit/'checkpoint-mismatch-audit').glob('*-indexed.json'))
    add('Checkpoint mismatch diagnosis', mismatch_state,
        f'{len(mismatch_rows)} failed checkpoints replayed; no projected controls recomputed')
    gap_folder = audit/'post-history-controls-v1'
    gap_status = read(audit/'post-history-controls-driver/status.json')
    gap_rows = [read(path) for path in (gap_folder/'cases').glob('*.json')]
    gap_state = str(gap_status.get('state','not started')).upper()
    if gap_state == 'RUNNING' and time.time()-gap_status.get('heartbeat_epoch',0)>45:
        gap_state = 'STALE / CHECK LOG'
    add('Post-history snapshot controls', gap_state,
        f'{len(gap_rows)}/7 attempted; {sum(bool(row.get("success")) for row in gap_rows)} successful; '
        'isolated variant, previously unavailable controls only; failures remain listed')
    for variant in ('core','indexed','indexed-retry-v1'):
        folder = audit/'recovered-policy-completion'/variant
        tracked = read(folder.parent/('indexed-retry-driver' if variant=='indexed-retry-v1' else variant+'-driver')/'status.json')
        manifest = read(folder/'manifest.json')
        progress = read(folder/'progress.json')
        total = len(manifest.get('selected_cases', []))
        done = len(list((folder/'cases').glob('*/complete.json')))
        errors = len(list((folder/'cases').glob('*/error.json')))
        state = str(tracked.get('state','not started')).upper()
        if state == 'RUNNING' and time.time()-tracked.get('heartbeat_epoch',0)>45:
            state = 'STALE / CHECK LOG'
        if (folder/'pass-complete.json').exists():
            state = 'PASS DONE / ERRORS' if errors else 'DONE'
        add(variant.title()+' recovered policy completion', state,
            f'{done}/{total or "?"} case-variants complete; {errors} errors; '
            f'cache {progress.get("cache", {})}; frozen recovered-control source')
    try:
        output = subprocess.check_output(['ps', '-eo', 'pid=,comm=,args='], text=True)
        processes = []
        for line in output.splitlines():
            fields = line.strip().split(None, 2)
            if len(fields) != 3 or 'python' not in fields[1]:
                continue
            pid, _, command = fields
            if int(pid) == os.getpid() or 'show_all_progress' in command:
                continue
            if any(name in command for name in ('revision_eval.', 'evaluate_recommendations.py',
                                                 'performSimulations.py', 'recover_control_only.py', 'recover.py')):
                name = next((token for token in command.split() if token.startswith('revision_eval.') or
                             token.endswith(('.py',))), 'python evaluation')
                processes.append(dict(pid=int(pid), task=name))
    except (OSError, subprocess.SubprocessError):
        processes = []
    return dict(at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                runner=runner, tasks=rows, active_cases=active,
                active_baseline=baseline_active, visible_processes=processes)

def render(data):
    print('Journal evaluation progress —', data['at'])
    print('Current chain:', data['runner'].get('stage', '?'), '/', data['runner'].get('state', '?'))
    print('Stress funding: saved legacy profiles; contains the historical wallet-unit issue.')
    print('Counts are saved work, not automatically valid paper results. ETA is for the current pass only.\n')
    for row in data['tasks']:
        print(f"{row['task']:<43} {row['state']:<23} {row['detail']}")
    for row in data['active_cases']:
        print(f"  Agent worker {row.get('pid')}: case {row.get('index')} {row.get('variant')}, elapsed {duration(time.time()-row.get('started_epoch', time.time()))}")
    if data['active_baseline']:
        row = data['active_baseline']
        print(f"  Baseline: case {row.get('index')} {row.get('variant')}, elapsed {duration(time.time()-row.get('started_epoch', time.time()))}")
    heartbeat = data['runner'].get('heartbeat_epoch')
    if heartbeat:
        print('Runner heartbeat age:', duration(time.time()-heartbeat))
    fallback = ('not visible from this shell; a tracked job heartbeat is fresh'
                if any(row['state'] == 'RUNNING' for row in data['tasks'])
                else 'none visible')
    print('Visible evaluation processes:', ', '.join(f"{r['pid']}:{r['task']}" for r in data['visible_processes']) or fallback)
    if data['runner'].get('log'):
        print('Current log:', data['runner']['log'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--watch', type=float, default=0, metavar='SECONDS')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.watch < 0:
        parser.error('--watch must be nonnegative')
    try:
        while True:
            data = collect(args.root)
            if args.watch and os.isatty(1):
                print('\033[2J\033[H', end='')
            print(json.dumps(data, indent=2)) if args.json else render(data)
            if not args.watch:
                break
            time.sleep(max(1, args.watch))
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
