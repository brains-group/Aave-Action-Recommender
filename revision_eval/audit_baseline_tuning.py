"""Read-only proposal/cache audit for the eight-setting static-HF plan.

Never imports a simulation driver or invokes replay. Uses only saved local packs,
JSON profiles, read-only indexed evidence and exact historical cache keys. Outputs
are an audit, not new prevention results. Existing data/caches remain untouched.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import types
from collections import Counter
from functools import lru_cache
from pathlib import Path

from revision_eval.policies import propose

GRID = tuple((trigger, target) for trigger in (1.05, 1.10, 1.25)
             for target in (1.20, 1.40, 1.60) if trigger < target)
POLICIES = ('static_hf', 'repay_only', 'deposit_only', 'automation_style')
STRIP_FIELDS = ('id', 'transactionHash', 'blockNumber', 'logIndex', 'pool',
                'asset_address', 'on_behalf_of', 'payer', 'recipient',
                'rate_mode', 'generation_error')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def partition(user, seed='static-hf-tuning-20261012', calibration_fraction=.8):
    """One deterministic partition per account, shared by all its checkpoints."""
    number = int(hashlib.sha256((seed + ':' + user.lower()).encode()).hexdigest(), 16)
    return 'calibration' if number < int(calibration_fraction * 2**256) else 'held_out'


def canonical(args):
    return pickle.dumps({k: v for k, v in args.items() if k != 'output_file'}, protocol=4)


def cache_key(rec, suffix, args, identity):
    return cache_key_digest(rec, suffix, hashlib.sha256(canonical(args)).hexdigest(), identity)


def cache_key_digest(rec, suffix, digest, identity):
    return Path(identity) / digest[:16] / (
        f"{rec['user']}_{int(rec['timestamp'])}_{suffix}.pkl")


def lookup(rec, suffix, args, identity, roots):
    relative = cache_key(rec, suffix, args, identity)
    return next((str(root / relative) for root in roots if (root / relative).is_file()), None)


def signature(proposal):
    return (proposal['action'], proposal.get('asset'), proposal.get('amount', 0.))


def recommendation(base, proposal, state, *, correct_type=True):
    new = dict(base)
    for field in STRIP_FIELDS:
        new.pop(field, None)
    new.update(case_id=state['case_id'], abstained=proposal['action'] is None)
    new['Index Event'] = (proposal['action'] or 'Deposit').lower()
    if correct_type:
        # The frozen converter reads `type`, not `Index Event`.
        new['type'] = new['Index Event']
    new['amount'] = proposal.get('amount', 0.)
    new['amountUSD'] = proposal['capital_usd']
    if proposal['action']:
        new['reserve'] = new['symbol'] = proposal['asset']
        new['priceInUSD'] = state['assets'][proposal['asset']]['price_usd']
    new.update(logAmount=math.log1p(new['amount']), logAmountUSD=math.log1p(new['amountUSD']))
    return new


def load_definition(path, name, namespace):
    """Execute an isolated pure definition, never its parent module imports."""
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def frozen_converter(source):
    # Extract constants and the converter only; importing profile generation would
    # import the protocol, logger and price machinery unnecessarily.
    utils = source / 'Aave-Simulator/simulator/utils.py'
    ns = {}
    tree = ast.parse(utils.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                (t.id.startswith('ACTION_') or t.id == 'TRANSACTION_TYPE_TO_ACTION_MAP')
                for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(utils), 'exec'), ns)
    path = source / 'Aave-Simulator/profile_gen/user_profile_generator.py'
    fn = load_definition(path, '_row_to_transaction', {'Dict': dict, 'Optional': __import__('typing').Optional})
    adapter = types.SimpleNamespace(ACTION_MAPPING=ns['TRANSACTION_TYPE_TO_ACTION_MAP'])
    return lambda row: fn(adapter, row)


def frozen_identity(source):
    path = source / 'utils/simulations.py'
    ns = dict(Path=Path, aave_sim_path=str(source / 'Aave-Simulator'), __file__=str(path))
    return load_definition(path, 'simulation_code_identity', ns)()


def indexed_adapter(source, coverage, assets):
    # Only these pure evidence modules are permitted. Bypass analysis/__init__,
    # which imports other analysis tools. All SQLite access uses mode=ro.
    package = types.ModuleType('analysis')
    package.__path__ = [str(source / 'Aave-Simulator/analysis')]
    sys.modules['analysis'] = package
    for name in ('asset_registry', 'indexed_history'):
        path = source / 'Aave-Simulator/analysis' / (name + '.py')
        spec = importlib.util.spec_from_file_location('analysis.' + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    path = source / 'utils/indexed_checkpoint.py'
    spec = importlib.util.spec_from_file_location('audit_frozen_indexed_checkpoint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ['AAVE_INDEXED_COVERAGE'] = str(coverage)
    os.environ['AAVE_INDEXED_ASSETS'] = str(assets)
    os.environ.pop('AAVE_HISTORICAL_EVIDENCE', None)
    return module.attach_checkpoint


@lru_cache(maxsize=32)
def profile_file(profiles, user):
    for rel in ('non_liquidated_profiles/profiles', 'liquidated_profiles/profiles', ''):
        path = profiles / rel / f'user_{user}.json'
        if path.is_file():
            return path, json.loads(path.read_text()), sha(path)
    raise FileNotFoundError('No stored profile for ' + user)


def limited_profile(profiles, rec, variant, attach):
    path, original, digest = profile_file(profiles, rec['user'])
    # Filtering and anchoring only replace top-level fields; keep the cached
    # original read-only and avoid copying every unused future transaction.
    profile = dict(original)
    cutoff = rec['timestamp'] - 600
    txs = [tx for tx in profile.get('transactions', []) if isinstance(tx, dict)]
    profile['transactions'] = [tx for tx in txs if tx.get('timestamp', 0) <= cutoff]
    future = sorted((tx for tx in txs if tx.get('timestamp', 0) > cutoff),
                    key=lambda tx: tx.get('timestamp', 0))
    horizon = max(1, int(future[-1]['timestamp'] - cutoff)) * 2 if (
        future and future[-1].get('timestamp')) else 604800
    if variant == 'indexed':
        attach(profile)
    return profile, horizon, str(path), digest


def intervention_args(rec, profile, horizon, control, convert, adjust):
    if rec['abstained']:
        return dict(profile=profile, lookahead_seconds=horizon), None, None
    rec = adjust(rec, control)
    if rec is None:
        return None, None, 'repayment_rejected_by_checkpoint_guard'
    tx = convert(rec)
    if tx is None:
        return None, None, 'transaction_conversion_failed'
    tx['timestamp'] = rec['timestamp']
    new = copy.deepcopy(profile)
    new['transactions'].append(tx)
    # Preserve the historical evaluator's numeric types and its horizon rule.
    cutoff = rec['timestamp'] - 600
    followup = max(1, horizon - (rec['timestamp'] - cutoff) * 2)
    return dict(profile=new, lookahead_seconds=followup), tx, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, default=Path('outputs/revision-audits-20260929/matched-results.json'))
    p.add_argument('--source', type=Path, default=Path('/home/spadef/.codex/sessions/2026/09/08/audit-evaluation-20260914/source'))
    p.add_argument('--output', type=Path, default=Path('outputs/baseline-tuning-cache-audit-20261012'))
    p.add_argument('--seed', default='static-hf-tuning-20261012')
    p.add_argument('--calibration-fraction', type=float, default=.8)
    a = p.parse_args()
    if not 0 < a.calibration_fraction < 1:
        p.error('Calibration fraction must be between zero and one')
    if a.output.exists():
        p.error('Use a fresh audit output directory; existing audit outputs are preserved')
    a.output.mkdir(parents=True)
    source = a.source.resolve()
    started = time.monotonic()
    index = json.loads(a.index.read_text())
    entries = [e for e in index['entries'] if e['study'] == 'existing_cohort']
    if {e['variant'] for e in entries} != {'core', 'indexed'}:
        raise ValueError('Expected existing main cohort only, both reconstructions')
    receipts = {str(a.index): sha(a.index)}
    origins = set()
    cohorts = []
    for entry in entries:
        cases, summary = Path(entry['source_cases']), Path(entry['source_summary'])
        for file, field in ((cases, 'source_cases_sha256'), (summary, 'source_summary_sha256')):
            receipts[str(file)] = sha(file)
            assert receipts[str(file)] == entry[field], 'Matched evidence changed: ' + str(file)
        reviewed = [json.loads(line) for line in cases.open()]
        matched = {row['case_id']: row for row in reviewed if row['status'] == 'common'}
        assert len(matched) == entry['common_cases']
        ledger = cases.parent.with_name('existing-merged-' + entry['variant'] + '-v1') / 'cases.jsonl'
        receipts[str(ledger)] = sha(ledger)
        assert receipts[str(ledger)] == json.loads(summary.read_text())['source_ledger_sha256']
        raw = {row['case_id']: row for row in map(json.loads, ledger.open()) if row['case_id'] in matched}
        assert len(raw) == len(matched)
        manifest = json.loads((ledger.parent / 'manifest.json').read_text())
        parent, original, recovery = [Path(manifest[k]) for k in ('parent', 'original', 'recovery')]
        for folder in (parent, original):
            receipts[str(folder / 'manifest.json')] = sha(folder / 'manifest.json')
            key = 'parent_manifest_sha256' if folder == parent else 'original_manifest_sha256'
            assert receipts[str(folder / 'manifest.json')] == manifest[key]
        origins.update(Path(path).parents[2] for row in raw.values() for path in row['sources'].values())
        cohorts.append((entry, matched, raw))
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    identity = frozen_identity(source)
    assert identity == parent_manifest['code_identity'], 'Frozen source or price file changed'
    inputs = original.parent / 'refreshed-recommendations-20260911'
    coverage, assets = inputs / 'inputs/coverage.json', inputs / 'inputs/assets.json'
    for file, field in ((inputs / 'recommendations.pkl', 'recommendations_sha256'),
                        (coverage, 'coverage_sha256'), (assets, 'assets_sha256')):
        receipts[str(file)] = sha(file)
        assert receipts[str(file)] == parent_manifest[field], 'Frozen input changed: ' + str(file)
    for folder in sorted(origins):
        file = folder / 'manifest.json'
        if file.exists():
            receipts[str(file)] = sha(file)
            manifest = json.loads(file.read_text())
            assert manifest['code_identity'] == identity
            assert manifest['recommendations_sha256'] == parent_manifest['recommendations_sha256']
    roots = [parent / 'cache/simulation_results']
    roots += [folder / 'simulations' for folder in sorted(origins)]
    roots += [folder / 'simulations' for folder in sorted(recovery.iterdir()) if folder.is_dir()]
    roots = list(dict.fromkeys(root.resolve() for root in roots if root.is_dir()))
    profiles = (source / 'profiles').resolve()
    attach = indexed_adapter(source, coverage, assets)
    convert = frozen_converter(source)
    # Extract the actual repayment clamp; no hand-written approximation of it.
    import numpy as np
    logger = types.SimpleNamespace(warning=lambda *args: None)
    ns = dict(np=np, logger=logger)
    load_definition(source / 'utils/simulations.py', 'updateAmountOrUSD', ns)
    adjust = load_definition(source / 'utils/simulations.py', 'update_recommendation_if_necessary', ns)
    stats = {}
    accounts = {'calibration': set(), 'held_out': set()}
    issues = Counter()
    proposals_file = (a.output / 'proposals.jsonl').open('w')
    problems_file = (a.output / 'execution-field-conflicts.jsonl').open('w')
    failures_file = (a.output / 'input-verification-failures.jsonl').open('w')
    total = sum(len(matched) for _, matched, _ in cohorts)
    done = 0
    for entry, matched, raw in cohorts:
        variant = entry['variant']
        vstats = dict(cases=0, controls=Counter(), reference=Counter(),
                      legacy_policies={policy: Counter() for policy in POLICIES},
                      configurations={}, partitions={})
        stats[variant] = vstats
        for case_id, row in raw.items():
            file = Path(row['sources']['static_hf'])
            packfile = file.parent / 'recommendations.pkl'
            with packfile.open('rb') as stream:
                pack = pickle.load(stream)
            state = pack['state']
            base = pack['recommendations']['static_hf'][0]
            user = str(base['user'])
            split = partition(user, a.seed, a.calibration_fraction)
            accounts[split].add(user.lower())
            vstats['partitions'].setdefault(split, Counter())['cases'] += 1
            funds = sum(max(0, asset['wallet']) * asset['price_usd'] for asset in state['assets'].values())
            record = dict(case_id=case_id, user=user, variant=variant, partition=split,
                          pack=str(packfile), pack_sha256=sha(packfile), proposals=[])
            reference = propose(state, 'static_hf', trigger=1.1, target=1.2, budget_usd=funds)
            vstats['reference']['proposal_identical' if signature(reference) == signature(pack['proposals']['static_hf']) else 'proposal_changed'] += 1
            # Count all four policy adapters against the actual frozen converter.
            for policy in POLICIES:
                sourcefile = Path(row['sources'][policy])
                pp = pack if sourcefile.parent == packfile.parent else pickle.load((sourcefile.parent / 'recommendations.pkl').open('rb'))
                savedrec = pp['recommendations'][policy][0]
                intended = pp['proposals'][policy]['action']
                if intended is None:
                    vstats['legacy_policies'][policy]['abstentions'] += 1
                    continue
                vstats['legacy_policies'][policy]['active'] += 1
                tx = convert(savedrec)
                actual = tx.get('action') if tx else None
                if actual != intended:
                    vstats['legacy_policies'][policy]['action_field_conflicts'] += 1
                    problems_file.write(json.dumps(dict(case_id=case_id, variant=variant, policy=policy,
                        source=str(sourcefile), proposed_action=intended, converted_action=actual,
                        type=savedrec.get('type'), index_event=savedrec.get('Index Event')), default=str) + '\n')
            control = pickle.load(file.open('rb'))['arms']['without']
            try:
                profile, horizon, path, digest = limited_profile(profiles, base, variant, attach)
                args = dict(profile=profile, lookahead_seconds=horizon)
                control_encoded = canonical(args)
                control_digest = hashlib.sha256(control_encoded).hexdigest()
                hit = lookup(base, 'without', args, identity, roots)
                record.update(profile=path, profile_sha256=digest, control_key=str(cache_key(base, 'without', args, identity)), control_cache=hit)
                vstats['controls']['exact_cache_hit' if hit else 'exact_cache_missing'] += 1
                if hit:
                    exact_control = pickle.load(Path(hit).open('rb'))
                    # Detect whether the saved pack was generated against the same
                    # checkpoint used by this full-input matching control.
                    wallet = exact_control['checkpoint_state']['wallet_balances']
                    if any(abs(asset['wallet'] - wallet.get(symbol, 0.)) > 1e-8 * max(1., abs(asset['wallet']))
                           for symbol, asset in state['assets'].items()):
                        issues['pack_wallet_differs_from_exact_control'] += 1
                        record['pack_wallet_mismatch'] = True
                    control = exact_control
            except Exception as error:
                profile, horizon, hit = None, None, None
                record['input_error'] = str(error)
                vstats['controls']['input_reconstruction_failed'] += 1
                failures_file.write(json.dumps(dict(case_id=case_id, variant=variant, error=str(error))) + '\n')
            distinct, new_distinct, checked = set(), set(), {}
            for trigger, target in GRID:
                name = f'{trigger:.2f}/{target:.2f}'
                counts = vstats['configurations'].setdefault(name, Counter())
                proposal = propose(state, 'static_hf', trigger=trigger, target=target, budget_usd=funds)
                rec = recommendation(base, proposal, state)
                detail = dict(configuration=name, **{k: proposal.get(k) for k in ('action', 'asset', 'amount', 'capital_usd', 'reason', 'target_reached')})
                counts['cases'] += 1
                counts['proposed_capital_usd'] += proposal['capital_usd']
                counts['active' if proposal['action'] else 'abstentions'] += 1
                status = None
                if profile is None:
                    status = 'input_reconstruction_failed'
                elif proposal['action'] is None:
                    # Identical no-action input: directly reuse the verified
                    # control. Do not hash/scan it another eight times per case.
                    detail.update(transaction=None, input_sha256=control_digest,
                        cache_key=str(cache_key_digest(rec, 'with', control_digest, identity)), cache=hit)
                    status = 'reuse_exact_control' if hit and not record.get('pack_wallet_mismatch') else 'uncertified_control_or_checkpoint'
                else:
                    intervention, transaction, error = intervention_args(rec, profile, horizon, control, convert, adjust)
                    detail['transaction'] = transaction
                    if error:
                        status = error
                    else:
                        transaction_key = pickle.dumps((transaction, intervention['lookahead_seconds']), protocol=4)
                        if transaction_key not in checked:
                            encoded = canonical(intervention)
                            full_digest = hashlib.sha256(encoded).hexdigest()
                            cached = lookup(rec, 'with', intervention, identity, roots)
                            if not hit or record.get('pack_wallet_mismatch'):
                                status = 'uncertified_control_or_checkpoint'
                            elif encoded == control_encoded:
                                status = 'reuse_exact_control'
                            elif cached:
                                value = pickle.load(Path(cached).open('rb'))
                                if not all(k in value for k in ('checkpoint_state', 'liquidation_stats', 'final_state')):
                                    raise ValueError('Invalid exact cache result: ' + cached)
                                status = 'reuse_exact_intervention'
                            else:
                                status = 'new_intervention_required'
                            checked[transaction_key] = full_digest, cached, status
                        full_digest, cached, status = checked[transaction_key]
                        detail['input_sha256'] = full_digest
                        detail['cache_key'] = str(cache_key_digest(rec, 'with', full_digest, identity))
                        detail['cache'] = cached
                        if proposal['action']:
                            distinct.add(full_digest)
                            if status == 'new_intervention_required':
                                new_distinct.add(full_digest)
                detail['audit_status'] = status
                counts[status] += 1
                record['proposals'].append(detail)
            vstats['partitions'][split]['distinct_active_inputs'] += len(distinct)
            vstats['partitions'][split]['distinct_new_interventions'] += len(new_distinct)
            vstats['cases'] += 1
            proposals_file.write(json.dumps(record, default=str, allow_nan=False) + '\n')
            done += 1
            if done % 500 == 0 or done == total:
                progress = dict(completed=done, total=total, seconds=time.monotonic() - started,
                                phase='offline proposal and exact input/cache audit', simulations_started=0)
                (a.output / 'progress.json').write_text(json.dumps(progress, indent=2))
                print(f"{done}/{total} checkpoint packs audited", flush=True)
    for stream in (proposals_file, problems_file, failures_file):
        stream.close()
    assert not accounts['calibration'] & accounts['held_out']
    result = dict(mode='offline_read_only', simulations_started=0, grid=GRID,
                  code_identity=identity, current_policy_sha256=sha(Path(__file__).with_name('policies.py')),
                  audit_script_sha256=sha(__file__), inputs=receipts, cache_roots=list(map(str, roots)),
                  seed=a.seed, calibration_fraction=a.calibration_fraction,
                  accounts={k: len(v) for k, v in accounts.items()}, account_overlap=0,
                  variants=stats, issues=issues, seconds=time.monotonic() - started,
                  scope='Only the already matched existing main cases; no stress outcomes read; no replay or simulator changes.',
                  limitation='Cache hits certify historical frozen inputs/code, not independent on-chain correctness. Changed code or funding/horizon invalidates them.')
    for file in (a.output / 'proposals.jsonl', a.output / 'execution-field-conflicts.jsonl', a.output / 'input-verification-failures.jsonl'):
        result.setdefault('output_sha256', {})[file.name] = sha(file)
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print('Complete: ' + str(a.output / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
