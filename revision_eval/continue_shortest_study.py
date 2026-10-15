"""Durably resume evaluation of completed stress recommendations, without regeneration."""
import argparse
import fcntl
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/high-risk-shortest-20260925')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.run.resolve()
    previous = json.loads((root/'status.json').read_text())
    study_key = previous['study_key']
    study = root/'studies'/study_key
    generation = study/'generation-core'
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    complete = json.loads((generation/'complete.json').read_text())
    selected = json.loads((root/'cohort/manifest.json').read_text())['selected']
    if complete != {'cases': selected, 'errors': 0}:
        raise ValueError('Generation must be complete and error-free')
    rec_path = generation/'recommendations.pkl'
    with rec_path.open('rb') as stream:
        recommendations = pickle.load(stream)
    if len(recommendations) != selected:
        raise ValueError('Exported recommendation count differs from cohort')
    case_ids = []
    for path in sorted((generation/'cases').glob('*.pkl')):
        with path.open('rb') as stream:
            case = pickle.load(stream)
        if case.get('error'):
            raise ValueError(f'Generation error in {path.name}')
        case_ids.append(case['case_id'])
    if len(case_ids) != selected or set(case_ids) != set(recommendations):
        raise ValueError('Case IDs differ between generation files and export')
    receipt = study/'evaluation-resume.json'
    if not receipt.exists():
        write(receipt, dict(previous_status=previous, validated_cases=selected,
                           recommendation_sha256=hashlib.sha256(rec_path.read_bytes()).hexdigest(),
                           funding='legacy saved profile wallets; affected by historical USD/token bug',
                           generation='completed artifacts reused; no recommendation rerun',
                           old_exit_cause='not established; artifacts complete but shell chain failed'))
    horizon = json.loads((generation/'manifest.json').read_text())['horizon_seconds']
    inputs = Path('/data/craft-soc/data/evaluation/refreshed-recommendations-20260911/inputs')
    common = ['--recommendations', str(rec_path), '--coverage', str(inputs/'coverage.json'),
              '--assets', str(inputs/'assets.json'), '--horizon-seconds', str(horizon)]
    evaluation = study/'agent-evaluation'
    baseline = study/'policy-baselines'
    stages = [
        ('agent_evaluation', [sys.executable, '-u', '-m', 'revision_eval.run_shortest_study',
                             'evaluate', '--run-dir', str(evaluation), '--workers', str(args.workers), *common]),
        ('agent_summary', [sys.executable, '-u', 'summarize_recommendation_evaluation.py', str(evaluation)]),
        ('policy_baselines', [sys.executable, '-u', '-m', 'revision_eval.run_shortest_study',
                             'baselines', '--source', str(ROOT), '--parent-run', str(evaluation),
                             '--policies', str(ROOT/'revision_eval/policies.py'), '--run', str(baseline),
                             '--once', *common]),
    ]
    env = os.environ.copy()
    for name in ('AAVE_EVALUATION_CACHE', 'AAVE_SIMULATION_CACHE', 'AAVE_RECOMMENDATIONS_FILE',
                 'AAVE_CORE_TRANSACTIONS', 'AAVE_SURVIVAL_DATA', 'AAVE_INDEXED_COVERAGE',
                 'AAVE_INDEXED_ASSETS', 'AAVE_HISTORICAL_EVIDENCE'):
        env.pop(name, None)
    env['PYTHONUNBUFFERED'] = '1'
    env['MPLCONFIGDIR'] = str(study/'matplotlib-cache')
    for stage, command in stages:
        log_path = study/(stage+'.log')
        started = time.time()
        with log_path.open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            def status(state, **extra):
                write(root/'status.json', dict(stage=stage, state=state, study_key=study_key,
                      runner_pid=os.getpid(), child_pid=child.pid, started_epoch=started,
                      heartbeat_epoch=time.time(), log=str(log_path), command=command, **extra))
            status('running')
            try:
                while True:
                    try:
                        code = child.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        status('running')
            except BaseException:
                child.terminate()
                child.wait()
                status('interrupted')
                raise
            if code:
                status('failed', returncode=code)
                raise SystemExit(code)
            status('stage_complete', returncode=0)
    errors = len(list((baseline/'cases').glob('*/error.json')))
    unavailable = len(list((baseline/'cases').glob('*/unavailable.json')))
    write(root/'status.json', dict(stage='finished', state='complete_with_exclusions' if errors or unavailable else 'complete',
          study_key=study_key, heartbeat_epoch=time.time(), baseline_errors=errors,
          baseline_unavailable=unavailable))

if __name__ == '__main__':
    main()
