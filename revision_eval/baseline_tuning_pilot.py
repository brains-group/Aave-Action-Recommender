"""Prepare a deduplicated tuning/correction queue and time ten actual interventions.

Uses the frozen common evaluator and existing control caches. No training,
collection, cohort selection or cache migration. Old artifacts remain read-only.
"""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import pickle
import statistics
import sys
import threading
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

from . import audit_baseline_tuning as audit
from .policies import propose

SOURCE = Path('/home/spadef/.codex/sessions/2026/09/08/audit-evaluation-20260914/source')
DEFAULT_RUN = Path('outputs/baseline-tuning-pilot-20261012')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    if path.suffix == '.pkl':
        with tmp.open('wb') as stream:
            pickle.dump(value, stream, protocol=4)
    else:
        tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def frozen_helpers(source, summary):
    import numpy as np
    identity = audit.frozen_identity(source)
    if identity != summary['code_identity']:
        raise ValueError('Frozen code or price history changed')
    coverage = next(Path(p) for p in summary['inputs'] if p.endswith('/inputs/coverage.json'))
    assets = next(Path(p) for p in summary['inputs'] if p.endswith('/inputs/assets.json'))
    for path in (coverage, assets):
        if audit.sha(path) != summary['inputs'][str(path)]:
            raise ValueError('Indexed source manifest changed')
    attach = audit.indexed_adapter(source, coverage, assets)
    convert = audit.frozen_converter(source)
    ns = dict(np=np, logger=types.SimpleNamespace(warning=lambda *args: None))
    audit.load_definition(source / 'utils/simulations.py', 'updateAmountOrUSD', ns)
    adjust = audit.load_definition(source / 'utils/simulations.py', 'update_recommendation_if_necessary', ns)
    return identity, attach, convert, adjust


def context(row, source, identity, attach):
    packfile = Path(row['pack'])
    if audit.sha(packfile) != row['pack_sha256']:
        raise ValueError('Checkpoint pack changed: ' + row['case_id'])
    pack = pickle.load(packfile.open('rb'))
    base = pack['recommendations']['static_hf'][0]
    profile, horizon, path, digest = audit.limited_profile((source / 'profiles').resolve(), base, row['variant'], attach)
    if digest != row['profile_sha256']:
        raise ValueError('Profile changed: ' + row['case_id'])
    args = dict(profile=profile, lookahead_seconds=horizon)
    if str(audit.cache_key(base, 'without', args, identity)) != row['control_key']:
        raise ValueError('Control input changed: ' + row['case_id'])
    control_file = Path(row['control_cache'])
    control = pickle.load(control_file.open('rb'))
    original = audit.profile_file((source / 'profiles').resolve(), base['user'])[1]
    cutoff = base['timestamp'] - 600
    future = sorted((t for t in original.get('transactions', []) if isinstance(t, dict)
                     and t.get('timestamp', 0) > cutoff), key=lambda t: t.get('timestamp', 0))
    return dict(profile=profile, horizon=horizon, future=future, base=base,
                control=control, control_file=str(control_file), pack=pack)


def make_pair(job, ctx):
    if job['recipe']['kind'] == 'grid':
        state = ctx['pack']['state']
        funds = sum(max(0, a['wallet']) * a['price_usd'] for a in state['assets'].values())
        p = propose(state, 'static_hf', trigger=job['recipe']['trigger'],
                    target=job['recipe']['target'], budget_usd=funds)
        rec = audit.recommendation(ctx['base'], p, state)
        return rec, {'policy': 'static_hf', 'is_at_risk': p['action'] is not None}
    file = Path(job['recipe']['pack'])
    if audit.sha(file) != job['recipe']['pack_sha256']:
        raise ValueError('Reference proposal pack changed')
    pack = pickle.load(file.open('rb'))
    rec, info = copy.deepcopy(pack['recommendations'][job['recipe']['policy']])
    rec['type'] = str(rec['Index Event']).lower()
    if str(pack['proposals'][job['recipe']['policy']]['action']).lower() != rec['type']:
        raise ValueError('Reference proposal action disagrees with recommendation')
    return rec, info


def pending_job(row, detail, recipe, use):
    return dict(key=detail['cache_key'], case_id=row['case_id'], variant=row['variant'],
                partition=row['partition'], row=row, recipe=recipe, uses=[use],
                input_sha256=detail['input_sha256'])


def choose_pilot(jobs):
    """Five workload quintiles per reconstruction; never inspect outcomes."""
    pilot = []
    boundaries = {}
    for variant in ('core', 'indexed'):
        pool = sorted((j for j in jobs.values() if j['variant'] == variant and
                       j['partition'] == 'calibration' and any(u['kind'] == 'grid' for u in j['uses'])),
                      key=lambda j: (j['previous_pair_seconds'], j['key']))
        if len(pool) < 5:
            raise ValueError('Not enough pending calibration work for timing strata')
        borders = []
        for i in range(5):
            group = pool[i * len(pool) // 5:(i + 1) * len(pool) // 5]
            for job in group:
                job['timing_stratum'] = f'{variant}-{i}'
            middle = group[len(group) // 2]
            pilot.append(middle['key'])
            borders.append(group[-1]['previous_pair_seconds'])
        boundaries[variant] = borders
    for job in jobs.values():
        if 'timing_stratum' not in job:
            band = next((i for i, edge in enumerate(boundaries[job['variant']])
                         if job['previous_pair_seconds'] <= edge), 4)
            job['timing_stratum'] = f"{job['variant']}-{band}"
    return pilot


def workload(jobs, configuration):
    selected = set()
    for key, job in jobs.items():
        if any(u['kind'] == 'reference' or (u['kind'] == 'grid' and
               (job['partition'] == 'calibration' or u['configuration'] == configuration))
               for u in job['uses']):
            selected.add(key)
    return selected


def correction_budget(jobs, measured, configurations=None):
    """Cost of corrections and frozen-setting testing; no calibration split.

    This replaces the prospective account-split budget, not its historical
    receipts. Calibration on the earlier model-training period is additional
    work and must not use this queue's main-cohort outcomes.
    """
    if configurations is None:
        configurations = tuple(f'{trigger:.2f}/{target:.2f}' for trigger, target in audit.GRID)
    times = {r['stratum']: r['seconds'] for r in measured['timings']
             if not r['intervention_cache_hit']}
    completed = {r['key'] for r in measured['timings']}
    references = {key for key, job in jobs.items()
                  if any(u['kind'] == 'reference' for u in job['uses'])}

    def budget(keys):
        remaining = keys - completed
        return dict(distinct_jobs=len(keys), completed=len(keys & completed),
                    remaining_jobs=len(remaining),
                    remaining_serial_seconds=sum(times[jobs[k]['timing_stratum']]
                                                 for k in remaining))

    frozen = {}
    for name in configurations:
        grid = {key for key, job in jobs.items()
                if any(u['kind'] == 'grid' and u['configuration'] == name
                       for u in job['uses'])}
        frozen[name] = dict(budget(references | grid),
                            additional_inputs_beyond_corrections=len(grid - references))
    return dict(corrections_only=budget(references), frozen_setting_testing=frozen,
                includes_earlier_period_calibration=False,
                calibration='Earlier model-training period only; old account split superseded',
                outcome_extrapolation=False)


def prepare(args):
    run = args.run.resolve()
    if (run / 'queue.json').exists():
        raise ValueError('Queue already exists; use it or choose a fresh run directory')
    run.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.audit / 'summary.json').read_text())
    if audit.sha(args.audit / 'proposals.jsonl') != summary['output_sha256']['proposals.jsonl']:
        raise ValueError('Audit ledger changed')
    identity, attach, convert, adjust = frozen_helpers(args.source.resolve(), summary)
    rows = [json.loads(line) for line in (args.audit / 'proposals.jsonl').open()]
    jobs = {}
    for row in rows:
        for detail in row['proposals']:
            if detail['audit_status'] != 'new_intervention_required':
                continue
            trigger, target = map(float, detail['configuration'].split('/'))
            use = dict(kind='grid', configuration=detail['configuration'])
            if detail['cache_key'] in jobs:
                jobs[detail['cache_key']]['uses'].append(use)
            else:
                jobs[detail['cache_key']] = pending_job(row, detail,
                    dict(kind='grid', trigger=trigger, target=target), use)
    grid_keys = set(jobs)
    ledgers = {}
    for variant in ('core', 'indexed'):
        file = next(Path(p) for p in summary['inputs'] if
                    f'/existing-merged-{variant}-v1/cases.jsonl' in p)
        if audit.sha(file) != summary['inputs'][str(file)]:
            raise ValueError('Reference source ledger changed')
        ledgers[variant] = {r['case_id']: r for r in map(json.loads, file.open())}
    roots = list(map(Path, summary['cache_roots']))
    reference_counts = Counter()
    for row in rows:
        # All four saved policies have the same active checkpoint set, verified
        # in the preceding exhaustive audit. Only these need correction work.
        reference = next(d for d in row['proposals'] if d['configuration'] == '1.10/1.20')
        if reference['action'] is None:
            continue
        ctx = context(row, args.source.resolve(), identity, attach)
        for policy in audit.POLICIES:
            file = Path(ledgers[row['variant']][row['case_id']]['sources'][policy]).parent / 'recommendations.pkl'
            pack = pickle.load(file.open('rb'))
            if pack['proposals'][policy]['action'] is None:
                continue
            recipe = dict(kind='reference', policy=policy, pack=str(file), pack_sha256=audit.sha(file))
            stub = dict(recipe=recipe)
            rec, _ = make_pair(stub, ctx)
            kw, tx, error = audit.intervention_args(rec, ctx['profile'], ctx['horizon'], ctx['control'], convert, adjust)
            if error:
                raise ValueError(f'{row["case_id"]} {policy}: {error}')
            assert tx['action'].lower() == rec['type']
            key = str(audit.cache_key(rec, 'with', kw, identity))
            reference_counts['active'] += 1
            if audit.lookup(rec, 'with', kw, identity, roots):
                reference_counts['exact_existing_hit'] += 1
                continue
            reference_counts['uncached_requests'] += 1
            use = dict(kind='reference', policy=policy)
            if key in jobs:
                jobs[key]['uses'].append(use)
            else:
                detail = dict(cache_key=key, input_sha256=hashlib.sha256(audit.canonical(kw)).hexdigest())
                jobs[key] = pending_job(row, detail, recipe, use)
        save(run / 'preparation-progress.json', dict(phase='reference corrections',
             reference_active_requests=reference_counts['active'], distinct_pending=len(jobs), simulations_started=0))
    parent = Path(next(p for p in summary['inputs'] if p.endswith('/evaluation/manifest.json'))).parent
    elapsed = {}
    for job in jobs.values():
        case = job['case_id']
        if case not in elapsed:
            elapsed[case] = float(json.loads((parent / 'cases' / (case + '.json')).read_text())['elapsed'])
        job['previous_pair_seconds'] = elapsed[case]
    pilot = choose_pilot(jobs)
    names = [f'{t:.2f}/{g:.2f}' for t, g in summary['grid']]
    budgets = {name: len(workload(jobs, name)) for name in names}
    manifest = dict(source=str(args.source.resolve()), audit=str(args.audit.resolve()),
        code_identity=identity, audit_summary_sha256=audit.sha(args.audit / 'summary.json'),
        policies_sha256=audit.sha(Path(__file__).with_name('policies.py')),
        pilot_selection='One median workload per quintile of saved paired elapsed, five per reconstruction; calibration only; no outcome selection',
        controls='Existing exact controls only; never simulate a missing control',
        cache_roots=summary['cache_roots'], reference_counts=reference_counts,
        distinct_grid_jobs=len(grid_keys), reference_jobs_not_in_full_grid=len(set(jobs) - grid_keys),
        reference_distinct_uncached=len({key for key, j in jobs.items() if any(u['kind'] == 'reference' for u in j['uses'])}),
        combined_jobs_by_frozen_setting=budgets, pilot_keys=pilot,
        original_inputs=summary['inputs'], logging='discard verbose timestamp arrays; structured results and errors saved',
        network_requests=0, created=time.time())
    save(run / 'queue.json', jobs)
    manifest['queue_sha256'] = audit.sha(run / 'queue.json')
    save(run / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('distinct_grid_jobs', 'reference_jobs_not_in_full_grid',
        'reference_distinct_uncached', 'combined_jobs_by_frozen_setting', 'pilot_keys')}, indent=2), flush=True)


def run_pilot(args):
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    if audit.sha(run / 'queue.json') != manifest['queue_sha256']:
        raise ValueError('Queue changed')
    guard = (run / 'pilot.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = json.loads((run / 'queue.json').read_text())
    source = Path(manifest['source'])
    # Build contexts before importing the evaluator to avoid the audit adapter's
    # deliberately minimal analysis namespace interfering with driver imports.
    summary = json.loads((Path(manifest['audit']) / 'summary.json').read_text())
    identity, attach, convert, adjust = frozen_helpers(source, summary)
    prepared = {}
    for key in manifest['pilot_keys']:
        job = jobs[key]
        ctx = context(job['row'], source, identity, attach)
        pair = make_pair(job, ctx)
        expected, tx, error = audit.intervention_args(pair[0], ctx['profile'], ctx['horizon'], ctx['control'], convert, adjust)
        if error or str(audit.cache_key(pair[0], 'with', expected, identity)) != key:
            raise ValueError('Pilot input differs from audited queue: ' + job['case_id'])
        if hashlib.sha256(audit.canonical(expected)).hexdigest() != job['input_sha256']:
            raise ValueError('Pilot full input digest mismatch')
        prepared[key] = ctx, pair, expected
    for name in list(sys.modules):
        if name == 'analysis' or name.startswith('analysis.'):
            del sys.modules[name]
    for name in ('data', 'profiles'):
        link = run / name
        if not link.exists():
            link.symlink_to((source / name).resolve(), target_is_directory=True)
    os.chdir(run)
    for name in ('cache', 'simulations', 'locks', 'logs', 'results'):
        (run / name).mkdir(exist_ok=True)
    rec_file = next(p for p in manifest['original_inputs'] if p.endswith('/recommendations.pkl'))
    os.environ.update(AAVE_EVALUATION_CACHE=str(run / 'cache'), AAVE_SIMULATION_CACHE=str(run / 'simulations'),
                      AAVE_RECOMMENDATIONS_FILE=rec_file)
    for name in ('AAVE_SURVIVAL_DATA', 'AAVE_CORE_TRANSACTIONS', 'AAVE_HISTORICAL_EVIDENCE',
                 'AAVE_INDEXED_COVERAGE', 'AAVE_INDEXED_ASSETS'):
        os.environ.pop(name, None)
    sys.path.insert(0, str(source))
    import performSimulations as evaluator
    import utils.simulations as simulations
    from utils.logger import set_log_file
    set_log_file('pilot-errors.log', False, str(run / 'logs'), file_level='WARNING')
    if simulations.simulation_code_identity() != identity:
        raise ValueError('Actual evaluator identity differs from the audited one')
    # Shared price/template initialization is measured once, not once per job.
    from simulator.utils import setup_protocol
    warm_start = time.perf_counter()
    with open(os.devnull, 'w') as discard, contextlib.redirect_stdout(discard):
        setup_protocol()
    warmup_seconds = time.perf_counter() - warm_start
    roots = [run / 'simulations', *map(Path, manifest['cache_roots'])]
    progress = dict(phase='pilot', completed=0, total=10, active_case=None,
                    active_started=None, controls_reused=0, new_simulations=0,
                    failures=0, warmup_seconds=warmup_seconds, pid=os.getpid())
    lock = threading.Lock()
    stop = threading.Event()
    def heartbeat():
        while not stop.is_set():
            with lock:
                value = dict(progress)
                value['active_seconds'] = time.time() - value['active_started'] if value['active_started'] else None
                save(run / 'progress.json', value)
            stop.wait(5)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    timings = []
    try:
        for key in manifest['pilot_keys']:
            job = jobs[key]
            dest = run / 'results' / (hashlib.sha256(key.encode()).hexdigest() + '.pkl')
            if dest.exists():
                timings.append(pickle.load(dest.open('rb'))['timing'])
                with lock:
                    progress['completed'] += 1
                continue
            ctx, pair, expected = prepared[key]
            rec = pair[0]
            with lock:
                progress.update(active_case=job['case_id'], active_variant=job['variant'],
                                active_started=time.time(), timing_stratum=job['timing_stratum'])
            arms = {}
            durations = {}
            def profile_hook(r, return_extras=False):
                profile = copy.deepcopy(ctx['profile'])
                return (profile, r['timestamp'], r['timestamp'] - 600, r['user'], ctx['future'], ctx['horizon']) if return_extras else profile
            def cache_hook(r, suffix, **kw):
                wanted = dict(profile=ctx['profile'], lookahead_seconds=ctx['horizon']) if suffix == 'without' else expected
                if audit.canonical(kw) != audit.canonical(wanted):
                    raise ValueError('Actual evaluator changed prepared simulation inputs')
                rel = audit.cache_key(r, suffix, kw, identity)
                if suffix == 'with' and str(rel) != key:
                    raise ValueError('Actual intervention key differs from queue')
                mutex = (run / 'locks' / (hashlib.sha256(str(rel).encode()).hexdigest() + '.lock')).open('a')
                with mutex:
                    fcntl.flock(mutex, fcntl.LOCK_EX)
                    hit = audit.lookup(r, suffix, kw, identity, roots)
                    if hit:
                        value = pickle.load(Path(hit).open('rb'))
                        durations[suffix] = 0.
                        durations[suffix + '_cache_hit'] = True
                    elif suffix == 'without':
                        raise ValueError('Missing exact control; refusing control replay')
                    else:
                        started = time.perf_counter()
                        value = simulations.run_simulation(**kw)
                        durations[suffix] = time.perf_counter() - started
                        durations[suffix + '_cache_hit'] = False
                        save(run / 'simulations' / rel, value)
                        with lock:
                            progress['new_simulations'] += 1
                    arms[suffix] = value
                    if suffix == 'without':
                        with lock:
                            progress['controls_reused'] += 1
                    return value
            evaluator.get_limited_user_profile = profile_hook
            evaluator.get_simulation_outcome = cache_hook
            evaluator.outputFile = str(run / 'logs/simulation.log')
            start = time.perf_counter()
            try:
                with open(os.devnull, 'w') as discard, contextlib.redirect_stdout(discard), contextlib.redirect_stderr(discard):
                    result = evaluator.process_recommendation(copy.deepcopy(pair))
                if not result.get('success') or set(arms) != {'without', 'with'}:
                    raise ValueError('Evaluator failed: ' + str(result.get('error', result)))
                timing = dict(key=key, case_id=job['case_id'], variant=job['variant'],
                    stratum=job['timing_stratum'], seconds=time.perf_counter() - start,
                    simulation_seconds=durations['with'], intervention_cache_hit=durations['with_cache_hit'])
                save(dest, dict(timing=timing, recommendation=pair, result=result, arms=arms,
                                transaction=expected['profile']['transactions'][-1]))
                timings.append(timing)
                print(f"{len(timings)}/10 {job['case_id']}: {timing['seconds']:.2f}s", flush=True)
                with lock:
                    progress['completed'] += 1
            except Exception as error:
                import traceback
                save(run / 'failure.json', dict(key=key, error=str(error), traceback=traceback.format_exc()))
                with lock:
                    progress.update(phase='failed', failures=progress['failures'] + 1)
                raise
        save(run / 'timings.json', dict(warmup_seconds=warmup_seconds, timings=timings))
        with lock:
            progress.update(phase='complete', active_case=None, active_started=None)
    finally:
        stop.set()
        thread.join(timeout=6)
        with lock:
            save(run / 'progress.json', progress)


def estimate(args):
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    jobs = json.loads((run / 'queue.json').read_text())
    measured = json.loads((run / 'timings.json').read_text())
    times = {r['stratum']: r['seconds'] for r in measured['timings'] if not r['intervention_cache_hit']}
    if len(times) != 10:
        raise ValueError('Need one newly simulated timing for each of the ten strata')
    configs = manifest['combined_jobs_by_frozen_setting']
    estimates = {}
    completed = {r['key'] for r in measured['timings']}
    for name in configs:
        selected = workload(jobs, name)
        strata = Counter(jobs[key]['timing_stratum'] for key in selected)
        remaining_strata = Counter(jobs[key]['timing_stratum'] for key in selected - completed)
        estimates[name] = dict(distinct_jobs=len(selected), new_jobs_remaining=len(selected - completed),
            total_serial_seconds=measured['warmup_seconds'] + sum(n * times[s] for s, n in strata.items()),
            remaining_serial_seconds=sum(n * times[s] for s, n in remaining_strata.items()),
            work_by_stratum=strata)
    cal_keys = {key for key, job in jobs.items() if job['partition'] == 'calibration' and any(u['kind'] == 'grid' for u in job['uses'])}
    result = dict(method='Workload-stratified pilot: one actual timing per quintile per reconstruction; weighted by pending job counts; single worker',
        caveat='Ten timings provide a planning estimate, not a statistical confidence interval. Cached old paired elapsed is used only for strata; actual pilot wall times determine the estimate.',
        calibration_grid_new_jobs=len(cal_keys),
        calibration_grid_serial_seconds=sum(times[jobs[key]['timing_stratum']] for key in cal_keys),
        pilot_completed=len(completed), warmup_seconds=measured['warmup_seconds'],
        pilot_seconds_range=[min(times.values()), max(times.values())],
        corrected_reference_jobs=manifest['reference_distinct_uncached'], configurations=estimates,
        original_controls_replayed=0, code_identity=manifest['code_identity'])
    save(run / 'runtime-estimate.json', result)
    save(run / 'correction-and-frozen-testing-budget.json', correction_budget(jobs, measured))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'run', 'estimate'))
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--audit', type=Path, default=Path('outputs/baseline-tuning-cache-audit-20261012'))
    p.add_argument('--source', type=Path, default=SOURCE)
    args = p.parse_args()
    {'prepare': prepare, 'run': run_pilot, 'estimate': estimate}[args.mode](args)


if __name__ == '__main__':
    main()
