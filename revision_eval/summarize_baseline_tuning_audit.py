"""Summarize the completed offline audit into a distinct-projection work budget."""
import argparse
import json
import pickle
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from revision_eval.audit_baseline_tuning import sha, partition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=Path('outputs/baseline-tuning-cache-audit-20261012'))
    p.add_argument('--report', type=Path, default=Path('docs/revision-eval/BASELINE_TUNING_CACHE_AUDIT_20261012.md'))
    a = p.parse_args()
    summary = json.loads((a.audit / 'summary.json').read_text())
    for name, digest in summary['output_sha256'].items():
        assert sha(a.audit / name) == digest, 'Audit output changed: ' + name
    names = [f'{t:.2f}/{g:.2f}' for t, g in summary['grid']]
    new = defaultdict(set)
    config_new = defaultdict(set)
    counts = defaultdict(Counter)
    certified = Counter()
    records = []
    active_cases = set()
    with (a.audit / 'proposals.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row['partition'] == partition(row['user'], summary['seed'], summary['calibration_fraction'])
            assert {detail['configuration'] for detail in row['proposals']} == set(names)
            assert len(row['proposals']) == len(names)
            records.append((row['case_id'], row['variant'], row['partition'], row['user']))
            if row.get('control_cache') and not row.get('pack_wallet_mismatch'):
                certified[row['variant']] += 1
            for detail in row['proposals']:
                if detail.get('transaction'):
                    assert detail['transaction']['action'] == detail['action'], 'Prospective action-type conflict'
                key = (row['variant'], row['partition'])
                counts[key][detail['audit_status']] += 1
                if detail['audit_status'] == 'new_intervention_required':
                    active_cases.add(row['case_id'])
                    new[key].add(detail['cache_key'])
                    config_new[key + (detail['configuration'],)].add(detail['cache_key'])
    assert len(records) == sum(s['cases'] for s in summary['variants'].values())
    assert len({case for case, _, _, _ in records}) == len(records)
    actual_conflicts = Counter()
    for line in (a.audit / 'execution-field-conflicts.jsonl').open():
        conflict = json.loads(line)
        actual_conflicts[conflict['variant'], conflict['policy']] += 1
    for variant, s in summary['variants'].items():
        assert sum(r[1] == variant for r in records) == s['cases']
        source_cases = next(Path(file) for file in summary['inputs']
                            if f'/existing-reviewed-{variant}-' in file and file.endswith('/cases.jsonl'))
        expected_cases = {r['case_id'] for r in map(json.loads, source_cases.open()) if r['status'] == 'common'}
        assert {r[0] for r in records if r[1] == variant} == expected_cases
        assert sum(sum(c.values()) for key, c in counts.items() if key[0] == variant) == s['cases'] * len(names)
        for policy, legacy in s['legacy_policies'].items():
            assert actual_conflicts[variant, policy] == legacy.get('action_field_conflicts', 0), 'Conflict inventory does not match summary'
    splits = defaultdict(set)
    for _, _, portion, user in records:
        splits[user.lower()].add(portion)
    assert all(len(values) == 1 for values in splits.values())
    budget = dict(calibration_new=sum(len(new[v, 'calibration']) for v in summary['variants']),
                  held_out_all_grid_new=sum(len(new[v, 'held_out']) for v in summary['variants']),
                  held_out_one_frozen_setting={name: sum(len(config_new[v, 'held_out', name]) for v in summary['variants']) for name in names},
                  full_grid_new=sum(map(len, new.values())),
                  certified_controls=dict(certified), account_partition_overlap=0,
                  simulations_started=0)
    budget['calibration_then_one_held_out_upper_bound'] = budget['calibration_new'] + max(budget['held_out_one_frozen_setting'].values())
    rec_file = next(Path(file) for file in summary['inputs'] if file.endswith('/recommendations.pkl'))
    assert sha(rec_file) == summary['inputs'][str(rec_file)]
    original_recs = pickle.load(rec_file.open('rb'))
    agent_fields = Counter()
    for rec, _ in original_recs.values():
        rec = rec.to_dict() if hasattr(rec, 'to_dict') else rec
        agent_fields[str(rec.get('Index Event', '')).lower(), str(rec.get('type', '')).lower()] += 1
    budget['agent_action_field_check'] = dict(total=len(original_recs),
        conflicts=sum(n for (event, execution_type), n in agent_fields.items() if event != execution_type),
        field_pairs=[dict(index_event=event, execution_type=typ, count=n) for (event, typ), n in agent_fields.items()])
    parent = Path(next(file for file in summary['inputs'] if file.endswith('/evaluation/manifest.json'))).parent
    pair_times = []
    for case in sorted(active_cases):
        saved = json.loads((parent / 'cases' / (case + '.json')).read_text())
        if isinstance(saved.get('elapsed'), (int, float)):
            pair_times.append(saved['elapsed'])
    if pair_times:
        budget['historical_pair_elapsed_seconds'] = dict(cases=len(pair_times),
            minimum=min(pair_times), median=statistics.median(pair_times), maximum=max(pair_times),
            meaning='Original paired agent case elapsed, including preparation/cache activity and both arms; not standalone new-intervention time.')
    (a.audit / 'work-budget.json').write_text(json.dumps(budget, indent=2))
    lines = [
        '# Bounded static-HF tuning: offline proposal and cache audit', '',
        'Completed 12 October 2026. No simulations, model training, downloads or network requests were started. Existing inputs and caches were read only. The manuscript and cover letter were not changed.', '',
        '## Scope and reproducibility', '',
        f"The audit uses only the already matched main-cohort cases: **{summary['variants']['core']['cases']:,} Core and {summary['variants']['indexed']['cases']:,} Indexed**. It does not select new checkpoints or read stress-test outcomes. Core and Indexed retain their own reconstructed inputs and case intersections.", '',
        'The bounded grid contains eight static-HF trigger/target pairs. Proposals use each saved checkpoint’s original same-asset wallet funds, with the capital budget equal to the sum of those funds at saved prices. The default $1,000 policy cap is not used. All four existing policy adapters are checked for action-field conflicts, but only static-HF receives new tuning proposals.', '',
        f"Frozen evaluator/cache identity: `{summary['code_identity']}`. The audit recomputed this identity from the frozen simulator, utility code, registry and price file. Matched ledgers, source summaries, manifests, recommendations and indexed-input manifests were checked against their recorded SHA-256 hashes. Each output row records its checkpoint-pack hash, profile hash and complete simulation-input fingerprint. Receipts are in `summary.json`.", '',
        f"Account partition: SHA-256 of `{summary['seed']}:<lowercase account>`, with the first {100 * summary['calibration_fraction']:.0f}% of the hash range assigned to calibration. This partitions existing cases, rather than selecting a new cohort. All checkpoints and both reconstructions of an account remain together. There are {summary['accounts']['calibration']:,} calibration and {summary['accounts']['held_out']:,} held-out accounts, with zero overlap.", '',
        '## Exact reuse rules', '',
        'The historical key hashes the protocol-4 pickle of the complete `profile` and `lookahead_seconds`, excluding only the log-output path. The result lives under the frozen code identity, input fingerprint, account, recommendation timestamp and arm suffix. Funding, history, indexed anchors, transaction metadata, action, token, amount and horizon therefore participate in reuse. Matching only a case ID or proposed amount cannot establish a hit.', '',
        'The audit rebuilds profiles with the frozen history cutoff (recommendation timestamp minus 600 seconds), uses the frozen strictly historical indexed adapter with SQLite opened read-only, and applies the actual extracted repayment guard and transaction converter. No replay driver is imported. An abstention reuses an exact no-action control. A matching active-intervention cache must also deserialize as a simulation result. Absent or inconsistent controls are reported separately and are not certified for a new intervention.', '',
        '| Reconstruction | Matched cases | Exact control hits | Missing exact controls | Reconstruction errors |',
        '|---|---:|---:|---:|---:|',
    ]
    for variant, s in summary['variants'].items():
        c = s['controls']
        lines.append(f"| {variant.capitalize()} | {s['cases']:,} | {c.get('exact_cache_hit', 0):,} | {c.get('exact_cache_missing', 0):,} | {c.get('input_reconstruction_failed', 0):,} |")
    lines += ['', f"Additional checkpoint consistency issues: `{summary['issues']}`.", '',
              '## Proposal and cache counts', '',
              'These are proposed actions and cache requirements, **not prevention or worsening results**. Counts below include all matched main cases. Active projections are deduplicated across settings for the work budget.', '',
              '| Trigger / target | Core actions | Indexed actions | Exact intervention hits (Core / Indexed) | New intervention requests (Core / Indexed) | Uncertified or rejected requests |',
              '|---|---:|---:|---:|---:|---:|']
    for name in names:
        cs = [summary['variants'][v]['configurations'][name] for v in ('core', 'indexed')]
        unresolved = sum(sum(value for key, value in c.items() if key in (
            'input_reconstruction_failed', 'uncertified_control_or_checkpoint',
            'repayment_rejected_by_checkpoint_guard', 'transaction_conversion_failed')) for c in cs)
        lines.append(f"| {name} | {cs[0]['active']:,} | {cs[1]['active']:,} | {cs[0].get('reuse_exact_intervention', 0):,} / {cs[1].get('reuse_exact_intervention', 0):,} | {cs[0].get('new_intervention_required', 0):,} / {cs[1].get('new_intervention_required', 0):,} | {unresolved:,} |")
    identical = sum(s['reference'].get('proposal_identical', 0) for s in summary['variants'].values())
    changed = sum(s['reference'].get('proposal_changed', 0) for s in summary['variants'].values())
    lines += ['', f"Recomputing the original 1.10/1.20 configuration reproduces the proposed action, asset and exact amount for **{identical:,}** saved cases; **{changed:,}** have a different proposal. This reproduces proposals, not the legacy adapter’s execution type."]
    lines += ['', '## Distinct replay budget', '',
              '| Reconstruction | New calibration projections across eight settings | New held-out projections across eight settings |',
              '|---|---:|---:|']
    for variant in summary['variants']:
        lines.append(f"| {variant.capitalize()} | {len(new[variant, 'calibration']):,} | {len(new[variant, 'held_out']):,} |")
    lines += ['', f"The full-grid audit finds **{budget['full_grid_new']:,} distinct new interventions**. Calibration requires **{budget['calibration_new']:,}**. After a configuration is frozen, its held-out cost ranges from **{min(budget['held_out_one_frozen_setting'].values()):,} to {max(budget['held_out_one_frozen_setting'].values()):,}** new interventions. Thus calibration followed by one held-out setting needs at most **{budget['calibration_then_one_held_out_upper_bound']:,}** new projections under the verified inputs, rather than running eight settings on every held-out case.", '',
              'No additional controls are scheduled. If any control input fails certification, resolve that identity discrepancy before including it; the count of those requests is kept separate above.', '',
              '### Runtime and the next bounded step', '',
              'The completed original baseline run recorded 579,549 seconds (6.71 days), including checkpoint reconstruction, waiting, cache reads and 680 new simulations. Dividing that elapsed time by 680 would not estimate the runtime of an individual new projection. Saved outcomes do not record complete per-projection wall times. A reliable ETA therefore requires an approved timing pilot.', '',
              'A concrete next step is to run at most ten distinct pending calibration interventions from this existing queue, reusing their verified controls, recording full wall times and loading the frozen inputs once. Those jobs remain part of the full calibration queue; this does not create another evaluation cohort. No pilot has been launched. Freeze the selection objective and how the two reconstructions contribute to it before replay; the plan calls for prevention with zero worsening and proposed capital as the tie-breaker.', '',
              'For budget illustration only (not measured ETAs):', '',
              '| Assumed mean time per new projection | Calibration | Calibration plus maximum one-setting held-out work |',
              '|---|---:|---:|']
    for seconds in (10, 60, 300):
        lines.append(f"| {seconds} seconds | {budget['calibration_new'] * seconds / 3600:.2f} hours | {budget['calibration_then_one_held_out_upper_bound'] * seconds / 3600:.2f} hours |")
    if pair_times:
        lines += ['', f"For additional context, the original agent case files for {len(pair_times):,} checkpoints needing new tuning actions record paired-case elapsed times from {min(pair_times):.2f} to {max(pair_times):.2f} seconds, with a median of {statistics.median(pair_times):.2f} seconds. These timings include both arms and cache activity and are not standalone new-intervention timings; they do not justify promising the ten-second or one-minute scenario."]
    lines += ['', '## Action-type conflict in saved baseline adapters', '',
              'The saved baseline generator changes `Index Event` to the proposed repayment/deposit, but leaves the inherited `type` field unchanged. The frozen simulator converter chooses its action from `type`. Consequently, a proposed repayment can be converted to a deposit (or another inherited action). This is an adapter issue; it is not evidence that the protocol simulator’s repayment rule is wrong.', '',
              '| Policy | Core active proposals | Core action-field conflicts | Indexed active proposals | Indexed action-field conflicts |',
              '|---|---:|---:|---:|---:|']
    for policy in ('static_hf', 'repay_only', 'deposit_only', 'automation_style'):
        cs = [summary['variants'][v]['legacy_policies'][policy] for v in ('core', 'indexed')]
        lines.append(f"| {policy} | {cs[0].get('active', 0):,} | {cs[0].get('action_field_conflicts', 0):,} | {cs[1].get('active', 0):,} | {cs[1].get('action_field_conflicts', 0):,} |")
    lines += ['', 'A directly inspected matched example is `00030-core`: the static-HF pack proposes repayment of 4.337979015630708 USDT, while `type=deposit`. The saved intervention arm records a final Deposit action. The complete converted-action conflict inventory is in `execution-field-conflicts.jsonl`; the table counts conflicting proposals, not unique users or demonstrated outcome reversals.', '',
              f"The original agent recommendations were checked separately: {len(original_recs):,} records, with {budget['agent_action_field_check']['conflicts']:,} disagreements between the two action fields. The detected field conflict is introduced by the baseline adapter retaining the agent’s old execution type when proposing a different action; this finding does not show an action-field error in the saved agent recommendations.", '',
              'The audit’s prospective static-HF recommendations set both action fields consistently. Old simulations are reusable only if their actual full transaction input matches that intended action. The original adapter, simulator, caches and completed results were not modified. The effect on prevention cannot be established from proposal inspection. Correcting the production baseline adapter and evaluating affected actions remains a prerequisite to a defensible tuned-policy comparison; the original four-policy outcome counts should not be treated as verified execution of their named policies without that check.', '',
              '## Verification and handoff', '',
              f"All {len(records):,} matched checkpoint packs were accounted for, producing {len(records) * len(names):,} configuration records. Existing 1.10/1.20 proposal equality is reported in `summary.json`; this checks proposal reproduction, separately from the action-field conflict. Exact cache verification applies only to the recorded frozen code and input assumptions, not independent on-chain accuracy. Changing the evaluator, price series, funding or follow-up requires another identity audit.", '',
              'Focused tests passed for cache invalidation when funding/horizon/transaction/code changes, action-field consistency at the converter, and account-level partitioning. Final independent ledger checks verified case coverage, distinct work counts and account separation.', '',
              'Run from `/home/spadef/Aave-Action-Recommender`:', '', '```bash',
              'python -m pytest -q tests/test_baseline_tuning_audit.py',
              'python -m revision_eval.audit_baseline_tuning --output outputs/baseline-tuning-cache-audit-repeat',
              'python -m revision_eval.summarize_baseline_tuning_audit --audit outputs/baseline-tuning-cache-audit-repeat',
              '```', '',
              'The audit refuses to overwrite an existing audit directory. Local source/result pickles are trusted research artifacts; no untrusted downloads are deserialized.', '',
              f"Machine-readable evidence: `{a.audit}/summary.json`, `work-budget.json`, `proposals.jsonl`, `execution-field-conflicts.jsonl`, and `input-verification-failures.jsonl`. The earlier partial pass in the adjacent `-partial-before-optimization` directory is incomplete and is not used as evidence. The completed audit took {summary['seconds'] / 60:.2f} minutes.", '',
              'Pending: approval of a bounded replay pilot; correction of production proposal action fields; measured timing; calibration and frozen-setting held-out evaluation. No new results have been claimed or inserted into the paper.', '']
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text('\n'.join(lines))
    progress_path = a.audit / 'progress.json'
    progress = json.loads(progress_path.read_text())
    progress.update(phase='complete', completeness_checks_passed=True,
                    configuration_records=len(records) * len(names), report=str(a.report))
    progress_path.write_text(json.dumps(progress, indent=2))
    print(json.dumps(budget, indent=2))
    print('Report: ' + str(a.report))


if __name__ == '__main__':
    main()
