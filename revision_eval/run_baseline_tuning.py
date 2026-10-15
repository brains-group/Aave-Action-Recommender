"""Chronological eight-setting calibration and corrected existing baselines.

Process workers share exact-input locks and a new cache; historical roots are
read-only. The main evaluation population and frozen simulator are unchanged.
"""
import argparse
import concurrent.futures as futures
import contextlib
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

from . import audit_baseline_tuning as audit
from . import baseline_tuning_pilot as pilot
from .policies import propose

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'outputs/baseline-tuning-chronological-20261012'
PILOT = ROOT / 'outputs/baseline-tuning-pilot-20261012'
HORIZON = 30 * 86400
ENGINE = None


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    if path.suffix == '.pkl':
        temp.write_bytes(pickle.dumps(value, protocol=4))
    else:
        temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


def job_id(key):
    return hashlib.sha256(key.encode()).hexdigest()


def outcome(arm):
    return bool(arm['liquidation_stats']['liquidated'])


def completed_rapid_calibration_pair(result, arms):
    """A legacy reporting skip is not a failed replay or a prevented event."""
    if not result.get('skipped') or not str(result.get('error', '')).startswith('Skipping because liquidation too fast'):
        return False
    if set(arms) != {'without', 'with'}:
        return False
    control = arms['without']['liquidation_stats']
    elapsed = control.get('time_to_liquidation')
    return (bool(control['liquidated']) and elapsed is not None and 0 <= elapsed <= 600
            and bool(arms['with']['liquidation_stats']['liquidated']))


def prefix_funding(profile, cutoff, inferencer_class):
    """Keep the frozen inference behavior, with a version-compatible receipt."""
    import pandas as pd
    txs = [t for t in profile['transactions'] if float(t['timestamp']) <= cutoff]
    history = pd.DataFrame(txs).rename(columns={'action': 'type', 'symbol': 'reserve'})
    inferencer = inferencer_class(1.5)
    wallet = inferencer.normalize_wallet_amounts(inferencer.infer_wallet_hybrid(history, history))
    provenance = dict(method='Frozen hybrid inference from pre-decision transactions only',
                      safety_factor=1.5, cutoff=cutoff, prefix_transactions=len(txs))
    return wallet, provenance


def count_outcomes(records):
    result = dict(cases=0, control_liquidations=0, prevented=0, worsened=0,
                  proposed_capital_usd=0., execution_failures=0)
    for record in records:
        c, w = record['control'], record['intervention']
        result['cases'] += 1
        result['control_liquidations'] += bool(c)
        result['prevented'] += bool(c and not w)
        result['worsened'] += bool(not c and w)
        result['proposed_capital_usd'] += record.get('proposed_capital_usd', 0.)
        result['execution_failures'] += record.get('execution_failed', False)
    result['prevention_rate'] = (result['prevented'] / result['control_liquidations']
                                 if result['control_liquidations'] else None)
    return result


def choose_setting(calibration):
    """Frozen objective; no main-cohort data is accepted by this function."""
    scored = []
    for name, variants in calibration.items():
        if any(variants[v]['prevention_rate'] is None for v in ('core', 'indexed')):
            raise ValueError('Calibration has no control liquidation cases in a reconstruction; no setting selected')
        if any(variants[v]['worsened'] for v in ('core', 'indexed')):
            continue
        rate = sum(variants[v]['prevention_rate'] for v in ('core', 'indexed')) / 2
        capital = sum(variants[v]['proposed_capital_usd'] for v in ('core', 'indexed'))
        trigger, target = map(float, name.split('/'))
        scored.append(((-rate, capital, name != '1.10/1.20', trigger, target), name))
    if not scored:
        raise ValueError('No calibration setting satisfies zero worsening')
    return min(scored)[1]


def prepare(args):
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'manifest.json').exists():
        raise ValueError('Already prepared; resume with run')
    old = json.loads((PILOT / 'manifest.json').read_text())
    summary = json.loads((Path(old['audit']) / 'summary.json').read_text())
    if audit.sha(PILOT / 'queue.json') != old['queue_sha256']:
        raise ValueError('Historical audited queue changed')
    source = Path(old['source'])
    identity, attach, _, _ = pilot.frozen_helpers(source, summary)
    for name, digest in summary['inputs'].items():
        if audit.sha(name) != digest:
            raise ValueError('Historical input changed: ' + name)
    dates = Path('/data/craft-soc/data/evaluation/refreshed-recommendations-20260911/generation-cache/date_ranges.pkl')
    train, _ = pickle.load(dates.open('rb'))
    boundary = train[0].timestamp()
    latest_history = boundary - HORIZON - 600 - 1
    old_jobs = json.loads((PILOT / 'queue.json').read_text())
    corrections = {key: j for key, j in old_jobs.items()
                   if any(u['kind'] == 'reference' for u in j['uses'])}
    rows_path = Path(old['audit']) / 'proposals.jsonl'
    if audit.sha(rows_path) != summary['output_sha256']['proposals.jsonl']:
        raise ValueError('Historical proposal ledger changed')
    rows = [json.loads(line) for line in rows_path.open()]
    users = sorted({r['user'] for r in rows}, key=lambda u: hashlib.sha256(
        ('chronological-baseline-calibration-20261012:' + u.lower()).encode()).hexdigest())
    counts = Counter()
    reasons = Counter()
    selected = []
    for scanned, user in enumerate(users, 1):
        path, original, digest = audit.profile_file((source / 'profiles').resolve(), user)
        txs = sorted((t for t in original['transactions'] if isinstance(t, dict) and
                      float(t.get('timestamp', 0)) <= latest_history), key=lambda t: t['timestamp'])
        if len(txs) < 10:
            reasons['fewer_than_ten_earlier_actions'] += 1
            continue
        # Selection uses only pre-boundary history; future transactions and
        # observed/simulated liquidation outcomes never enter the selection.
        timestamp = float(txs[-1]['timestamp']) + 600
        if timestamp + HORIZON >= boundary:
            raise ValueError('Calibration follow-up crosses training boundary')
        for variant in ('core', 'indexed'):
            if args.calibration_cases and counts[variant] >= args.calibration_cases:
                continue
            profile = dict(original, transactions=txs)
            profile.pop('indexed_checkpoint', None)
            if variant == 'indexed':
                try:
                    attach(profile)
                except ValueError as error:
                    reasons['indexed:' + str(error)] += 1
                    continue
            case = dict(case_id=f'cal-{counts[variant]:03d}-{variant}', user=user,
                        variant=variant, timestamp=timestamp, cutoff=timestamp - 600,
                        profile=str(path), profile_sha256=digest, history_transactions=len(txs),
                        followup_end=timestamp + HORIZON)
            selected.append(case)
            counts[variant] += 1
        save(run / 'preparation-progress.json', dict(scanned_users=scanned,
             calibration_selected=dict(counts), selection_reasons=dict(reasons)))
        if args.calibration_cases and all(counts[v] == args.calibration_cases for v in ('core', 'indexed')):
            break
    if any(not counts[v] or (args.calibration_cases and counts[v] != args.calibration_cases)
           for v in ('core', 'indexed')):
        raise ValueError('Insufficient earlier profile/snapshot coverage: ' + str(counts))
    for name in ('data', 'profiles'):
        (run / name).symlink_to((source / name).resolve(), target_is_directory=True)
    if args.reuse_run:
        reuse = args.reuse_run.resolve()
        reuse_manifest = json.loads((reuse / 'manifest.json').read_text())
        for field, expected in (('code_identity', identity), ('calibration_boundary', boundary),
                                ('calibration_horizon_seconds', HORIZON)):
            if reuse_manifest[field] != expected:
                raise ValueError('Expanded-run input mismatch: ' + field)
        prior_cases = json.loads((reuse / 'calibration-cases.json').read_text())
        selected_map = {c['case_id']: c for c in selected}
        if any(selected_map.get(c['case_id']) != c for c in prior_cases):
            raise ValueError('Earlier calibration cases changed during expansion')
        for name in ('simulations', 'locks', 'results', 'reference-index', 'calibration', 'simulation-timings'):
            (reuse / name).mkdir(exist_ok=True)
            (run / name).symlink_to(reuse / name, target_is_directory=True)
    save(run / 'all-testing-jobs.json', old_jobs)
    save(run / 'correction-jobs.json', corrections)
    save(run / 'calibration-cases.json', selected)
    manifest = dict(source=str(source), audit=old['audit'], code_identity=identity,
        runner_sha256=audit.sha(__file__), policies_sha256=audit.sha(ROOT / 'revision_eval/policies.py'),
        original_inputs=summary['inputs'], cache_roots=[str(PILOT / 'simulations'), *old['cache_roots']],
        grid=[f'{t:.2f}/{g:.2f}' for t, g in audit.GRID], workers=args.workers,
        calibration_cases_per_variant=args.calibration_cases, calibration_boundary=boundary,
        date_ranges_sha256=audit.sha(dates), calibration_horizon_seconds=HORIZON,
        selection='SHA256 account order; last action with >=10 prefix actions and 30-day follow-up strictly before initial model cutoff; no outcome selection',
        funding='Calibration: prefix-only frozen WalletInferencer(1.5); main testing: unchanged saved wallets',
        objective='Zero worsening; equal-weight mean Core/Indexed prevention; lower capital; reference then numeric order for tie',
        correction_jobs=len(corrections), selected_cases=counts,
        hashes={p: audit.sha(run / p) for p in ('all-testing-jobs.json', 'correction-jobs.json', 'calibration-cases.json')},
        reuse_run=str(args.reuse_run.resolve()) if args.reuse_run else None,
        calibration_workers=args.calibration_workers or args.workers,
        created=time.time(), network_requests=0)
    save(run / 'manifest.json', manifest)
    print(json.dumps(dict(calibration_cases=counts, correction_inputs=len(corrections),
                          configurations=manifest['grid'], boundary=boundary)), flush=True)


class Worker:
    def __init__(self, run):
        self.run = Path(run)
        self.manifest = json.loads((self.run / 'manifest.json').read_text())
        self.source = Path(self.manifest['source'])
        summary = json.loads((Path(self.manifest['audit']) / 'summary.json').read_text())
        self.identity, original_attach, self.convert, self.adjust = pilot.frozen_helpers(self.source, summary)
        coverage = next(p for p in summary['inputs'] if p.endswith('/inputs/coverage.json'))
        assets = next(p for p in summary['inputs'] if p.endswith('/inputs/assets.json'))
        def attach(profile):
            os.environ.update(AAVE_INDEXED_COVERAGE=coverage, AAVE_INDEXED_ASSETS=assets)
            try:
                original_attach(profile)
            finally:
                os.environ.pop('AAVE_INDEXED_COVERAGE', None)
                os.environ.pop('AAVE_INDEXED_ASSETS', None)
        self.attach = attach
        for name in list(sys.modules):
            if name == 'analysis' or name.startswith('analysis.'):
                del sys.modules[name]
        workspace = self.run / 'workers' / str(os.getpid())
        workspace.mkdir(parents=True, exist_ok=True)
        for name in ('data', 'profiles'):
            link = workspace / name
            if not link.exists():
                link.symlink_to((self.source / name).resolve(), target_is_directory=True)
        os.chdir(workspace)
        os.environ.update(AAVE_EVALUATION_CACHE=str(workspace / 'cache'),
            AAVE_SIMULATION_CACHE=str(self.run / 'simulations'),
            MPLCONFIGDIR=str(workspace / 'matplotlib'),
            AAVE_RECOMMENDATIONS_FILE=next(p for p in summary['inputs'] if p.endswith('/recommendations.pkl')))
        for name in ('AAVE_SURVIVAL_DATA', 'AAVE_CORE_TRANSACTIONS', 'AAVE_HISTORICAL_EVIDENCE',
                     'AAVE_INDEXED_COVERAGE', 'AAVE_INDEXED_ASSETS'):
            os.environ.pop(name, None)
        sys.path.insert(0, str(self.source))
        import performSimulations
        import utils.simulations
        import tools.run_single_simulation
        from profile_gen.wallet_inference import WalletInferencer
        from utils.logger import set_log_file
        self.evaluator, self.simulations = performSimulations, utils.simulations
        self.runner, self.inferencer = tools.run_single_simulation, WalletInferencer
        set_log_file('worker.log', False, str(workspace), file_level='WARNING')
        if self.simulations.simulation_code_identity() != self.identity:
            raise ValueError('Actual evaluator identity differs')
        from simulator.utils import setup_protocol
        with open(os.devnull, 'w') as discard, contextlib.redirect_stdout(discard):
            setup_protocol()
        self.roots = [self.run / 'simulations', *map(Path, self.manifest['cache_roots'])]
        self.reference_ledgers = {}
        for variant in ('core', 'indexed'):
            path = next(Path(p) for p in summary['inputs'] if f'/existing-merged-{variant}-v1/cases.jsonl' in p)
            if not path.is_absolute():
                path = ROOT / path
            self.reference_ledgers[variant] = {r['case_id']: r for r in map(json.loads, path.open())}
        self.status = dict(worker=os.getpid(), phase='idle', new_simulations=0,
                           cache_hits=0, heartbeat=time.time())
        self.lock = threading.Lock()
        threading.Thread(target=self.heartbeat, daemon=True).start()

    def heartbeat(self):
        while True:
            with self.lock:
                self.status['heartbeat'] = time.time()
                save(self.run / 'active' / f'{os.getpid()}.json', self.status)
            time.sleep(3)

    def cached(self, rec, suffix, kwargs, allow_control=False):
        relative = audit.cache_key(rec, suffix, kwargs, self.identity)
        # The suffix labels the arm, not simulation inputs; one lock/cache
        # lookup for equivalent control and abstention inputs avoids duplication.
        lock_key = str(relative).replace('_without.pkl', '_with.pkl')
        mutex = self.run / 'locks' / (job_id(lock_key) + '.lock')
        mutex.parent.mkdir(exist_ok=True)
        with mutex.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            hit = audit.lookup(rec, suffix, kwargs, self.identity, self.roots)
            if not hit:
                other = 'with' if suffix == 'without' else 'without'
                hit = audit.lookup(rec, other, kwargs, self.identity, self.roots)
            if hit:
                with self.lock:
                    self.status['cache_hits'] += 1
                return pickle.load(Path(hit).open('rb'))
            if suffix == 'without' and not allow_control:
                raise ValueError('Missing exact main-cohort control; refusing replay')
            started = time.perf_counter()
            value = self.simulations.run_simulation(**kwargs)
            save(self.run / 'simulations' / relative, value)
            save(self.run / 'simulation-timings' / (job_id(lock_key) + '.json'),
                 dict(key=str(relative), seconds=time.perf_counter() - started,
                      worker=os.getpid(), stage=self.status['phase']))
            with self.lock:
                self.status['new_simulations'] += 1
            return value

    def evaluate(self, ctx, pair, expected_key=None, digest=None, allow_control=False):
        expected, tx, error = audit.intervention_args(pair[0], ctx['profile'], ctx['horizon'],
                    ctx['control'], self.convert, self.adjust)
        if error:
            raise ValueError(error)
        key = str(audit.cache_key(pair[0], 'with', expected, self.identity))
        if expected_key is not None and key != expected_key:
            raise ValueError('Intervention differs from audited key')
        if digest is not None and hashlib.sha256(audit.canonical(expected)).hexdigest() != digest:
            raise ValueError('Full intervention input differs from audit')
        arms = {}
        def profile_hook(rec, return_extras=False):
            profile = copy.deepcopy(ctx['profile'])
            return (profile, rec['timestamp'], rec['timestamp'] - 600, rec['user'],
                    ctx['future'], ctx['horizon']) if return_extras else profile
        def cache_hook(rec, suffix, **kwargs):
            wanted = dict(profile=ctx['profile'], lookahead_seconds=ctx['horizon']) if suffix == 'without' else expected
            if audit.canonical(kwargs) != audit.canonical(wanted):
                raise ValueError('Evaluator changed prepared simulation inputs')
            arms[suffix] = self.cached(rec, suffix, kwargs, allow_control)
            return arms[suffix]
        self.evaluator.get_limited_user_profile = profile_hook
        self.evaluator.get_simulation_outcome = cache_hook
        self.evaluator.outputFile = str(self.run / 'workers' / str(os.getpid()) / 'simulation.log')
        result = self.evaluator.process_recommendation(copy.deepcopy(pair))
        rapid_pair = allow_control and completed_rapid_calibration_pair(result, arms)
        if (not result.get('success') and not rapid_pair) or set(arms) != {'without', 'with'}:
            raise ValueError('Evaluator failed: ' + str(result.get('error', result)))
        if audit.canonical(dict(profile=ctx['profile'], lookahead_seconds=ctx['horizon'])) == audit.canonical(expected):
            tx = None
        failures = [f for f in arms['with'].get('execution_failures', [])
                    if tx is not None and f.get('timestamp') == pair[0]['timestamp']]
        return dict(key=key, result=result, arms=arms, transaction=tx,
                    evaluation_status='completed_rapid_pair_not_prevented' if rapid_pair else 'success',
                    control=outcome(arms['without']), intervention=outcome(arms['with']),
                    execution_failed=bool(failures), recommendation=pair)

    def main_job(self, job):
        dest = self.run / 'results' / (job_id(job['key']) + '.pkl')
        if dest.exists():
            return dict(kind='main', key=job['key'], reused=True)
        ctx = pilot.context(job['row'], self.source, self.identity, self.attach)
        pair = pilot.make_pair(job, ctx)
        value = self.evaluate(ctx, pair, job['key'], job['input_sha256'])
        # Record all corrected reference input keys, even existing-cache hits
        # omitted from the pending queue. Aggregation can then reuse them safely.
        if any(u['kind'] == 'reference' for u in job['uses']):
            index = {}
            ledger = self.reference_ledgers[job['variant']][job['case_id']]
            for policy in audit.POLICIES:
                path = Path(ledger['sources'][policy]).parent / 'recommendations.pkl'
                pack = pickle.load(path.open('rb'))
                rec, info = copy.deepcopy(pack['recommendations'][policy])
                rec['type'] = str(rec['Index Event']).lower()
                args, tx, err = audit.intervention_args(rec, ctx['profile'], ctx['horizon'], ctx['control'], self.convert, self.adjust)
                if err:
                    raise ValueError('Corrected reference index: ' + err)
                index[policy] = dict(key=str(audit.cache_key(rec, 'with', args, self.identity)),
                    capital_usd=pack['proposals'][policy]['capital_usd'], timestamp=rec['timestamp'],
                    action=pack['proposals'][policy]['action'],
                    applied_capital_usd=(float(tx['amount']) * ctx['pack']['state']['assets'][tx['symbol']]['price_usd']) if tx else 0.)
            save(self.run / 'reference-index' / (job['case_id'] + '.json'), index)
        save(dest, value)
        return dict(kind='main', key=job['key'], reused=False)

    def checkpoint(self, profile):
        class Stop(BaseException):
            pass
        state = {}
        original = self.runner.run_all_strategies
        def capture(**kw):
            sim = kw['sim']
            user = sim.get_user(kw['user_address'])
            account = sim.get_user_account_data(kw['user_address'])
            if user.get('stable_debt'):
                raise ValueError('Stable debt unavailable in existing policy adapter')
            state.update(account, state_timestamp=kw['base_timestamp'], assets={})
            state['weighted_collateral_usd'] = account['total_collateral_usd'] * account['effective_liquidation_threshold']
            for symbol in sorted(set(user['wallet']) | set(user['debt']) | set(user['collateral'])):
                state['assets'][symbol] = dict(price_usd=sim.prices[sim.price_key_for_user(user, symbol)],
                    wallet=user['wallet'].get(symbol, 0.),
                    debt=user['debt'].get(symbol, 0.) * sim.reserves[symbol].variable_borrow_index,
                    liquidation_threshold=sim.risk_parameters(user, symbol)[1],
                    collateral_enabled=sim.collateral_enabled(user, symbol))
            raise Stop()
        self.runner.run_all_strategies = capture
        try:
            try:
                self.runner.run_simulation(profile=copy.deepcopy(profile), lookahead_seconds=1)
            except Stop:
                pass
        finally:
            self.runner.run_all_strategies = original
        if not state:
            raise ValueError('Checkpoint capture failed')
        return state

    def calibration_job(self, case):
        folder = self.run / 'calibration' / case['case_id']
        if (folder / 'complete.json').exists():
            return dict(kind='calibration', case_id=case['case_id'], reused=True)
        if audit.sha(case['profile']) != case['profile_sha256']:
            raise ValueError('Calibration profile changed')
        profile = json.loads(Path(case['profile']).read_text())
        original_txs = profile['transactions']
        profile['transactions'] = [t for t in original_txs if float(t.get('timestamp', 0)) <= case['cutoff']]
        profile.pop('indexed_checkpoint', None)
        wallet, provenance = prefix_funding(profile, case['cutoff'], self.inferencer)
        profile['initial_wallet'] = wallet
        profile['wallet_inference'] = provenance
        if case['variant'] == 'indexed':
            self.attach(profile)
        state = self.checkpoint(profile)
        state.update(case_id=case['case_id'], timestamp=case['timestamp'])
        last = profile['transactions'][-1]
        base = dict(user=case['user'], timestamp=case['timestamp'], type='deposit',
                    **{'Index Event': 'deposit'}, reserve=last.get('symbol', 'USDC'),
                    symbol=last.get('symbol', 'USDC'), amount=0., amountUSD=0., priceInUSD=1.)
        horizon = HORIZON + 600  # Existing fixed-followup adapter convention.
        if case['timestamp'] + HORIZON >= self.manifest['calibration_boundary']:
            raise ValueError('Calibration future crosses boundary')
        args = dict(profile=profile, lookahead_seconds=horizon)
        control = self.cached(base, 'without', args, allow_control=True)
        for symbol, balance in control['checkpoint_state']['wallet_balances'].items():
            if abs(state['assets'][symbol]['wallet'] - balance) > 1e-8 * max(1, abs(balance)):
                raise ValueError('Calibration checkpoint wallet mismatch')
        ctx = dict(profile=profile, horizon=horizon, control=control,
            future=sorted((t for t in original_txs if case['cutoff'] < float(t.get('timestamp', 0)) < self.manifest['calibration_boundary']), key=lambda t: t['timestamp']))
        funds = sum(max(0, a['wallet']) * a['price_usd'] for a in state['assets'].values())
        proposal_records = []
        for trigger, target in audit.GRID:
            name = f'{trigger:.2f}/{target:.2f}'
            dest = folder / (name.replace('/', '-') + '.pkl')
            if dest.exists():
                continue
            p = propose(state, 'static_hf', trigger=trigger, target=target, budget_usd=funds)
            rec = audit.recommendation(base, p, state)
            value = self.evaluate(ctx, (rec, dict(is_at_risk=p['action'] is not None)), allow_control=True)
            value.update(configuration=name, case=case, proposal=p, proposed_capital_usd=p['capital_usd'])
            save(dest, value)
            proposal_records.append(dict(configuration=name, key=value['key'], action=p['action']))
        save(folder / 'state.pkl', dict(state=state, funding=provenance, profile_sha256=case['profile_sha256']))
        save(folder / 'complete.json', dict(case_id=case['case_id'], variants=8, proposals=proposal_records))
        return dict(kind='calibration', case_id=case['case_id'], reused=False)

    def work(self, item):
        kind, payload = item
        name = payload['case_id']
        with self.lock:
            self.status.update(phase=kind, case_id=name, started=time.time())
        started = time.perf_counter()
        try:
            with open(os.devnull, 'w') as discard, contextlib.redirect_stdout(discard), contextlib.redirect_stderr(discard):
                value = self.main_job(payload) if kind == 'main' else self.calibration_job(payload)
            value.update(seconds=time.perf_counter() - started, success=True)
        except Exception as error:
            value = dict(kind=kind, case_id=name, success=False, error=str(error), traceback=traceback.format_exc(),
                         seconds=time.perf_counter() - started)
            save(self.run / 'failures' / (job_id(payload.get('key', name)) + '.json'), value)
        finally:
            with self.lock:
                self.status.update(phase='idle', case_id=None, started=None)
        return value


def initialize(run):
    global ENGINE
    ENGINE = Worker(run)


def work(item):
    return ENGINE.work(item)


def run_batch(pool, tasks, run, stage, history):
    pending = {pool.submit(work, item): item for item in tasks}
    total = len(pending)
    completed = 0
    started = time.time()
    while pending:
        finished, _ = futures.wait(pending, timeout=5, return_when=futures.FIRST_COMPLETED)
        for future in finished:
            pending.pop(future)
            result = future.result()
            history.append(result)
            completed += 1
            print(f'{stage}: {completed}/{total} {result["kind"]} {result.get("case_id", "")} success={result["success"]} {result["seconds"]:.1f}s', flush=True)
        save(run / 'progress.json', dict(phase=stage, heartbeat=time.time(), started=started,
             completed_stage_tasks=completed, total_stage_tasks=total,
             remaining_stage_tasks=len(pending), failures=sum(not r['success'] for r in history),
             completed_calibration=sum(r['kind'] == 'calibration' and r['success'] for r in history),
             completed_main_inputs=sum(r['kind'] == 'main' and r['success'] for r in history),
             elapsed_seconds=time.time() - started,
             task_rate_eta_seconds=(time.time() - started) * len(pending) / completed if completed >= 4 else None))
    save(run / 'task-receipts.json', history)


def summarize_calibration(run, manifest):
    cases = json.loads((run / 'calibration-cases.json').read_text())
    summary = {}
    for config in manifest['grid']:
        by_variant = {}
        for variant in ('core', 'indexed'):
            records = []
            for case in cases:
                if case['variant'] != variant:
                    continue
                path = run / 'calibration' / case['case_id'] / (config.replace('/', '-') + '.pkl')
                if not path.exists():
                    raise ValueError('Missing calibration outcome: ' + str(path))
                records.append(pickle.load(path.open('rb')))
            by_variant[variant] = count_outcomes(records)
        summary[config] = by_variant
    save(run / 'calibration-summary.json', summary)
    name = choose_setting(summary)
    save(run / 'frozen-setting.json', dict(configuration=name, frozen=time.time(),
        calibration_summary_sha256=audit.sha(run / 'calibration-summary.json'),
        cases_sha256=manifest['hashes']['calibration-cases.json'], objective=manifest['objective']))
    return name


def aggregate_testing(run, manifest, setting):
    """Preserve the full matched case sets, including all abstaining cases."""
    summary_source = json.loads((Path(manifest['audit']) / 'summary.json').read_text())
    roots = [run / 'simulations', *map(Path, manifest['cache_roots'])]
    matched = {}
    for variant in ('core', 'indexed'):
        path = next(Path(p) for p in summary_source['inputs'] if f'/existing-reviewed-{variant}-v2/cases.jsonl' in p)
        matched[variant] = {r['case_id']: r for r in map(json.loads, path.open()) if r['status'] == 'common'}
    def read_arm(key, known=None):
        aliases = (key, key.replace('_with.pkl', '_without.pkl'))
        for relative in aliases:
            for root in roots:
                path = root / relative
                if path.exists():
                    return pickle.load(path.open('rb'))
        if known and Path(known).exists():
            return pickle.load(Path(known).open('rb'))
        raise ValueError('Missing completed intervention: ' + key)
    records = {v: [] for v in ('core', 'indexed')}
    for line in (Path(manifest['audit']) / 'proposals.jsonl').open():
        row = json.loads(line)
        variant, case_id = row['variant'], row['case_id']
        old = matched[variant][case_id]
        control = old['parent']['control']
        methods = {'agent': dict(control=control, intervention=old['parent']['intervention'])}
        reference = next(p for p in row['proposals'] if p['configuration'] == '1.10/1.20')
        if reference['action'] is None:
            methods.update({p: dict(control=control, intervention=control, proposed_capital_usd=0.,
                                    execution_failed=False, attempted=False) for p in audit.POLICIES})
        else:
            index = json.loads((run / 'reference-index' / (case_id + '.json')).read_text())
            for policy in audit.POLICIES:
                meta = index[policy]
                arm = read_arm(meta['key'])
                failed = any(f.get('timestamp') == meta['timestamp'] for f in arm.get('execution_failures', []))
                methods[policy] = dict(control=control, intervention=outcome(arm),
                    proposed_capital_usd=meta['capital_usd'], execution_failed=failed,
                    executed_capital_usd=0. if failed else meta['applied_capital_usd'], attempted=meta['action'] is not None)
        detail = next(p for p in row['proposals'] if p['configuration'] == setting)
        if detail['action'] is None:
            intervention, failed = control, False
        else:
            arm = read_arm(detail['cache_key'], detail.get('cache'))
            timestamp = float(Path(row['control_key']).stem.split('_')[-2])
            intervention = outcome(arm)
            failed = any(f.get('timestamp') == timestamp for f in arm.get('execution_failures', []))
        methods['tuned_static_hf'] = dict(control=control, intervention=intervention,
            proposed_capital_usd=detail['capital_usd'], execution_failed=failed, attempted=detail['action'] is not None)
        records[variant].append(dict(case_id=case_id, user=row['user'], configuration=setting, methods=methods))
    final = {}
    for variant, rows in records.items():
        if len(rows) != len(matched[variant]):
            raise ValueError('Final population differs from frozen matched cases')
        final[variant] = {policy: count_outcomes([r['methods'][policy] for r in rows])
                          for policy in ('agent', *audit.POLICIES, 'tuned_static_hf')}
        path = run / (variant + '-matched-cases.jsonl')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        temporary.replace(path)
    save(run / 'matched-results.json', dict(configuration=setting, by_variant=final,
        interpretation='Paired simulated liquidation eligibility, not observed execution',
        calibration_summary_sha256=audit.sha(run / 'calibration-summary.json'),
        code_identity=manifest['code_identity']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'run'))
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--calibration-cases', type=int, default=25)
    p.add_argument('--calibration-workers', type=int)
    p.add_argument('--reuse-run', type=Path)
    args = p.parse_args()
    if args.workers < 1 or args.calibration_cases < 0 or (args.calibration_workers is not None and args.calibration_workers < 1):
        p.error('workers must be positive; cases must be nonnegative (0 means all eligible)')
    if args.mode == 'prepare':
        return prepare(args)
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    if audit.sha(__file__) != manifest['runner_sha256'] or audit.sha(ROOT / 'revision_eval/policies.py') != manifest['policies_sha256']:
        raise ValueError('Runner/policies changed; retain frozen run or use fresh directory')
    for name, digest in manifest['hashes'].items():
        if audit.sha(run / name) != digest:
            raise ValueError('Frozen queue changed')
    guard = (run / 'run.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('results', 'reference-index', 'active', 'simulations', 'locks', 'failures'):
        (run / name).mkdir(exist_ok=True)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    history = []
    corrections = list(json.loads((run / 'correction-jobs.json').read_text()).values())
    calibration = json.loads((run / 'calibration-cases.json').read_text())
    if manifest.get('reuse_run'):
        # The original four-worker coordinator finishes reference corrections.
        # Extra calibration workers share its cache/locks but never duplicate
        # its pending correction tasks. Final testing waits for those outputs.
        tasks = [('calibration', case) for case in calibration]
        try:
            with futures.ProcessPoolExecutor(max_workers=manifest['calibration_workers'],
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(str(run),)) as pool:
                run_batch(pool, tasks, run, 'expanded_calibration', history)
            if any(not r['success'] for r in history):
                raise ValueError('Expanded calibration tasks failed; see failures')
            setting = summarize_calibration(run, manifest)
            while True:
                remaining = sum(not (run / 'results' / (job_id(j['key']) + '.pkl')).exists() for j in corrections)
                previous_active = False
                with (Path(manifest['reuse_run']) / 'run.lock').open('a') as previous_guard:
                    try:
                        fcntl.flock(previous_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        previous_active = True
                if not remaining and not previous_active:
                    break
                save(run / 'progress.json', dict(phase='waiting_for_original_corrections', heartbeat=time.time(),
                    configuration=setting, remaining_correction_inputs=remaining,
                    original_coordinator_active=previous_active, failures=0))
                time.sleep(10)
            jobs = json.loads((run / 'all-testing-jobs.json').read_text())
            selected = [('main', j) for j in jobs.values()
                        if any(u['kind'] == 'grid' and u['configuration'] == setting for u in j['uses'])
                        and not (run / 'results' / (job_id(j['key']) + '.pkl')).exists()]
            with futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(str(run),)) as pool:
                run_batch(pool, selected, run, 'winning_setting_testing', history)
            if any(not r['success'] for r in history):
                raise ValueError('Final testing tasks failed')
            save(run / 'progress.json', dict(phase='aggregating', heartbeat=time.time(), configuration=setting, failures=0))
            aggregate_testing(run, manifest, setting)
            save(run / 'progress.json', dict(phase='complete', heartbeat=time.time(), configuration=setting,
                                            failures=0, results='matched-results.json'))
        except Exception as error:
            save(run / 'progress.json', dict(phase='needs_attention', heartbeat=time.time(), error=str(error),
                                            failures=sum(not r['success'] for r in history)))
            raise
        return
    # Interleave stages so corrections and calibration both make progress.
    tasks = []
    for i in range(max(len(corrections), len(calibration))):
        if i < len(calibration):
            tasks.append(('calibration', calibration[i]))
        if i < len(corrections):
            tasks.append(('main', corrections[i]))
    try:
        with futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(str(run),)) as pool:
            run_batch(pool, tasks, run, 'calibration_and_corrections', history)
            if any(not r['success'] for r in history):
                raise ValueError('Failed tasks remain; see failures directory')
            setting = summarize_calibration(run, manifest)
            jobs = json.loads((run / 'all-testing-jobs.json').read_text())
            selected = [('main', j) for j in jobs.values()
                        if any(u['kind'] == 'grid' and u['configuration'] == setting for u in j['uses'])
                        and not (run / 'results' / (job_id(j['key']) + '.pkl')).exists()]
            run_batch(pool, selected, run, 'winning_setting_testing', history)
            if any(not r['success'] for r in history):
                raise ValueError('Failed testing tasks remain')
        save(run / 'progress.json', dict(phase='aggregating', heartbeat=time.time(), configuration=setting, failures=0))
        aggregate_testing(run, manifest, setting)
        save(run / 'progress.json', dict(phase='complete', heartbeat=time.time(),
             configuration=setting, failures=0, results='matched-results.json'))
    except Exception as error:
        save(run / 'progress.json', dict(phase='needs_attention', heartbeat=time.time(), error=str(error),
                                       failures=sum(not r['success'] for r in history)))
        raise


if __name__ == '__main__':
    main()
