"""Censor-aware metrics and divergence traces from frozen historical replay outputs.

No simulation is rerun. Indexed references share a source with snapshot anchors;
they are withheld evidence, not independent archive-node ground truth.
"""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def horizon_label(timestamp, observed_times, observation_end, horizon=604800):
    if any(timestamp < t <= timestamp + horizon for t in observed_times):
        return True
    if observation_end >= timestamp + horizon:
        return False
    return None


def metrics(rows):
    counts = dict(tp=0, fp=0, tn=0, fn=0)
    for row in rows:
        y, p = row['label'], row['flag']
        counts['tp' if y and p else 'fn' if y else 'fp' if p else 'tn'] += 1
    tp, fp, tn, fn = (counts[k] for k in ('tp','fp','tn','fn'))
    ratio = lambda n,d: n/d if d else None
    return dict(counts, n=len(rows), positive_prevalence=ratio(tp+fn,len(rows)),
                precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn),
                false_positive_rate=ratio(fp,fp+tn), false_negative_rate=ratio(fn,tp+fn))


def evaluate(traces, alignment, horizon):
    groups = collections.defaultdict(list)
    for row in alignment:
        groups[row['account']].append(row)
    ends = {u:max(r['profile_transaction']['timestamp'] for r in rs) for u,rs in groups.items()}
    liquidation_times = {u:[r['profile_transaction']['timestamp'] for r in rs
                           if r['profile_transaction']['action']=='Liquidated'] for u,rs in groups.items()}
    holdout = set()
    for rows in groups.values():
        ordered = sorted(rows,key=lambda r:(int(r['core']['blockNumber']),int(r['core']['logIndex']),r['core']['id']))
        holdout.update(r['checkpoint_id'] for r in ordered[int(len(ordered)*.8):])
    if len({r['checkpoint_id'] for r in traces}) != len(traces):
        raise ValueError('Duplicate replay checkpoint IDs')
    result, examples = {}, {}
    for split in ('all_diagnostic','final_20_percent'):
        selected=[r for r in traces if split=='all_diagnostic' or r['checkpoint_id'] in holdout]
        for policy in ('strict','static','dynamic'):
            at_event=[dict(label=bool(r['observed']),flag=bool(r[policy])) for r in selected]
            prospective=[]; censored=0; lead=[]
            for r in selected:
                if r['observed']:continue
                label=horizon_label(r['timestamp'],liquidation_times[r['account']],ends[r['account']],horizon)
                if label is None:
                    censored+=1;continue
                prospective.append(dict(label=label,flag=bool(r[policy]),account=r['account']))
                if label and r[policy]:
                    lead.append(min(t-r['timestamp'] for t in liquidation_times[r['account']] if r['timestamp']<t<=r['timestamp']+horizon))
                kind='tp' if label and r[policy] else 'fn' if label else 'fp' if r[policy] else 'tn'
                if policy=='strict' and kind not in examples:
                    earlier=[x for x in traces if x['account']==r['account'] and x['execution_index']<=r['execution_index']]
                    examples[kind]=dict(checkpoint=r,lead_to_next_execution_seconds=lead[-1] if label and r[policy] else None,
                        first_zero_debt_observed_liquidation=next((x['checkpoint_id'] for x in earlier if x.get('zero_debt_observed_liquidation')),None),
                        interpretation='Pre-event state; observed execution label is not independently certified eligibility')
            accounts=collections.defaultdict(list)
            for r in prospective:accounts[r['account']].append(r)
            account_rows=[dict(label=any(r['label'] for r in rs),flag=any(r['flag'] for r in rs)) for rs in accounts.values()]
            result[split+':'+policy]=dict(at_event=metrics(at_event),seven_day=metrics(prospective),
                censored_non_liquidation_checkpoints=censored,
                account_level_any_flag_any_positive=metrics(account_rows),
                median_true_warning_lead_seconds=statistics.median(lead) if lead else None,
                warning_lead_unit='true-positive checkpoint to next execution; repeated warnings retained')
    return result,examples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    aligned_path=a.evidence/'aligned-checkpoints.json'
    alignment=json.loads(aligned_path.read_text());summary={};hashes={str(aligned_path):sha(aligned_path)}
    for variant in ('baseline-profile','transfers-profile','snapshots-profile'):
        source=a.evidence/(variant+'.json');raw=json.loads(source.read_text());hashes[str(source)]=sha(source)
        results,examples=evaluate(raw['traces'],alignment,604800)
        errors_path=a.evidence/(variant+'-state-errors.json');errors=json.loads(errors_path.read_text());hashes[str(errors_path)]=sha(errors_path)
        absolute=[r['absolute_usd_error'] for r in errors]
        relative=[r['relative_error'] for r in errors if r['relative_error'] is not None]
        summary[variant]=dict(metrics=results,failed_actions=len(raw['failures']),
            zero_debt_observed_liquidations=raw['zero_debt_observed_liquidations'],
            state_reference=dict(n=len(errors),mean_absolute_usd_error=statistics.mean(absolute),
                median_absolute_usd_error=statistics.median(absolute),median_relative_error=statistics.median(relative),
                independent_ground_truth=False),source_revision=raw.get('revision'))
        for item in examples.values():
            trace=item['checkpoint'];failures=[r for r in raw['failures'] if r['checkpoint_id'].startswith(trace['account']+':')]
            item['account_execution_failures']=failures
        (a.output/(variant+'-traces.json')).write_text(json.dumps(examples,indent=2))
    report=dict(variants=summary,horizon_seconds=604800,source_sha256=hashes,code_sha256=sha(Path(__file__)),
        status='completed_cached_diagnostic',independent_validation='BLOCKED: no independent historical account-state reference supplied',
        limitations=['Existing 198-account diagnostic cohort; no new cohort and no claim of population representativeness',
        'Final 20 percent is a diagnostic chronological slice per account, not a new independent model test',
        'Follow-up ends at last source-aligned profile event; horizon negatives assume completeness within retained histories',
        'Same-time future executions excluded from prospective labels; at-event counts are separate',
        'At-event negatives are non-liquidation actions, not certified healthy states',
        'Indexed post-block references share collection lineage with anchors; not independent on-chain validation',
        'Historical source vintage, wallets, aggregation limitations and simulator assumptions retained; not current corrected simulation outcomes',
        'Account any-flag/any-positive metrics measure co-occurrence across that account, not warning timeliness'])
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({name:value['metrics']['final_20_percent:strict']['seven_day'] for name,value in summary.items()},indent=2))

if __name__=='__main__':main()
