"""Summarize corrected reference baselines independently of calibration selection."""
import argparse
import json
import pickle
import time
from pathlib import Path

from . import audit_baseline_tuning as audit
from .run_baseline_tuning import DEFAULT_RUN, ROOT, count_outcomes, job_id, outcome, save


def summarize(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    source = json.loads((Path(manifest['audit']) / 'summary.json').read_text())
    originals = {}
    for variant in ('core', 'indexed'):
        path = next(Path(p) for p in source['inputs'] if f'/existing-reviewed-{variant}-v2/cases.jsonl' in p)
        path = path if path.is_absolute() else ROOT / path
        if audit.sha(path) != source['inputs'][str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)]:
            raise ValueError('Frozen matched-case ledger changed')
        originals[variant] = {r['case_id']: r for r in map(json.loads, path.open()) if r['status'] == 'common'}
    roots = [run / 'simulations', *map(Path, manifest['cache_roots'])]
    rows = {v: [] for v in ('core', 'indexed')}
    for line in (Path(manifest['audit']) / 'proposals.jsonl').open():
        audited = json.loads(line)
        variant, case = audited['variant'], audited['case_id']
        original = originals[variant][case]
        control = original['parent']['control']
        methods = {'agent': dict(control=control, intervention=original['parent']['intervention'])}
        active = next(p for p in audited['proposals'] if p['configuration'] == '1.10/1.20')['action'] is not None
        index = json.loads((run / 'reference-index' / (case + '.json')).read_text()) if active else {}
        for policy in audit.POLICIES:
            if not active:
                methods[policy] = dict(control=control, intervention=control, proposed_capital_usd=0., execution_failed=False)
                continue
            meta = index[policy]
            path = next((root / meta['key'] for root in roots if (root / meta['key']).exists()), None)
            if path is None:
                raise ValueError('Missing exact corrected intervention: ' + meta['key'])
            arm = pickle.load(path.open('rb'))
            failed = any(f.get('timestamp') == meta['timestamp'] for f in arm.get('execution_failures', []))
            methods[policy] = dict(control=control, intervention=outcome(arm), proposed_capital_usd=meta['capital_usd'],
                execution_failed=failed, executed_capital_usd=0. if failed else meta['applied_capital_usd'])
        rows[variant].append(dict(case_id=case, user=audited['user'], methods=methods))
    result = {}
    for variant, records in rows.items():
        if len(records) != len(originals[variant]):
            raise ValueError('Correction summary changed matched population')
        result[variant] = {p: count_outcomes([r['methods'][p] for r in records]) for p in ('agent', *audit.POLICIES)}
        path = run / f'corrected-baseline-{variant}-cases.jsonl'
        temp = path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(r) + '\n' for r in records))
        temp.replace(path)
    save(run / 'corrected-baseline-results.json', dict(by_variant=result,
        code_identity=manifest['code_identity'], configuration='Fixed references: trigger 1.10 / target 1.20',
        interpretation='Paired simulated liquidation eligibility; existing agent results and matched cases unchanged',
        independent_of_calibration_selection=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    keys = json.loads((run / 'correction-jobs.json').read_text())
    while True:
        completed = sum((run / 'results' / (job_id(key) + '.pkl')).exists() for key in keys)
        save(run / 'correction-summary-progress.json', dict(phase='waiting_for_corrections', heartbeat=time.time(),
                                                          completed=completed, total=len(keys)))
        if completed == len(keys):
            break
        if not args.wait:
            raise SystemExit(f'{completed}/{len(keys)} correction inputs complete')
        time.sleep(10)
    try:
        summarize(run)
        save(run / 'correction-summary-progress.json', dict(phase='complete', heartbeat=time.time(),
                                                          completed=completed, total=len(keys)))
    except Exception as error:
        save(run / 'correction-summary-progress.json', dict(phase='failed', heartbeat=time.time(), error=str(error)))
        raise


if __name__ == '__main__':
    main()
