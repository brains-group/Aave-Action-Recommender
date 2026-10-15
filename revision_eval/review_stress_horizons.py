"""Score saved corrected stress trajectories at 7/30 days; no simulator calls."""
import argparse,collections,json,math,pickle,time
from pathlib import Path
from . import audit_baseline_tuning as audit
ROOT=Path(__file__).resolve().parents[1]
CORRECTED=ROOT/'outputs/stress-agent-state-correction-20261013'
BOT=ROOT/'outputs/bayesian-tuned-bot-20261013'
RECS=ROOT/'outputs/high-risk-shortest-20260925/studies/fixed-2592000-seconds-trained-v1/generation-core/recommendations.pkl'
DEFAULT=ROOT/'outputs/stress-seven-day-review-20261014'


def by_deadline(flag,event_time,deadline):
    if not flag:
        if event_time is not None:raise ValueError('Unflagged arm has an event time')
        return False
    if event_time is None or not math.isfinite(float(event_time)):
        raise ValueError('Flagged arm missing a usable first-event timestamp')
    return float(event_time)<=deadline


def review(output):
    output.mkdir(parents=True,exist_ok=False)
    assert (CORRECTED/'verification.json').exists() and (BOT/'verification.json').exists()
    recs=pickle.load(RECS.open('rb'));rows=list(map(json.loads,(CORRECTED/'matched-cases.jsonl').open()))
    previous=json.loads((CORRECTED/'matched-results.json').read_text());case_rows=[];results={}
    assert len(rows)==1695
    for days in (7,30):
        counts={v:collections.Counter(cases=0,control_liquidations=0,agent_prevented=0,agent_worsened=0,bot_prevented=0,bot_worsened=0) for v in ('core','indexed')};misses={v:[] for v in counts}
        for row in rows:
            agent=row['methods']['agent'];bot=row['methods']['tuned_rule_bot'];decision=float(recs[row['case_id']][0]['timestamp']);deadline=decision+days*86400
            assert agent['control']==bot['control']
            control=by_deadline(agent['control'],agent['control_event_time'],deadline)
            action=by_deadline(agent['intervention'],agent['intervention_event_time'],deadline)
            # All new-bot intervention arms are unflagged over the full saved
            # thirty-day study, therefore also unflagged over a shorter window.
            if bot['intervention']:raise ValueError('Need bot event timestamp for flagged intervention')
            c=counts[row['variant']];c['cases']+=1;c['control_liquidations']+=control;c['agent_prevented']+=control and not action;c['agent_worsened']+=not control and action;c['bot_prevented']+=control
            if control and action:misses[row['variant']].append(dict(index=row['index'],case_id=row['case_id']))
            case_rows.append(dict(index=row['index'],variant=row['variant'],case_id=row['case_id'],days=days,decision=decision,deadline=deadline,control_event_time=agent['control_event_time'],agent_event_time=agent['intervention_event_time'],control_liquidates=control,agent_liquidates=action,bot_liquidates=False))
        for variant,c in counts.items():
            c=dict(c);n=c['control_liquidations'];c['agent_prevention_rate']=c['agent_prevented']/n;c['bot_prevention_rate']=c['bot_prevented']/n
            c['agent_remaining_cases']=misses[variant];counts[variant]=c
            if days==30:
                original=previous['by_variant'][variant]['agent'];assert c['control_liquidations']==original['control_liquidations'];assert c['agent_prevented']==original['prevented'];assert c['agent_worsened']==original['worsened']
        results[str(days)]=counts
    inputs={str(p):audit.sha(p) for p in [CORRECTED/'matched-cases.jsonl',CORRECTED/'matched-results.json',CORRECTED/'verification.json',BOT/'verification.json',RECS,Path(__file__)]}
    result=dict(by_days=results,definition='First recorded simulated liquidation eligibility by recommendation timestamp + days*86400 inclusive; same absolute deadline for both arms.',recommendations_unchanged=True,matched_case_set_unchanged=True,thirty_day_summary_reproduced=True,simulation_calls=0,inputs=inputs,created=time.time())
    (output/'summary.json').write_text(json.dumps(result,indent=2));(output/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in case_rows));print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args();review(a.output.resolve())
