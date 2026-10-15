"""Match completed stress agent and baseline policies with control-state checks.

Read-only. Every selected case/variant receives a ledger row. A shared denominator
requires successful agent and policy results, paired arms, source identity and
matching control states. No failed/unavailable outcome is treated as a negative.
"""
import argparse
import collections
import hashlib
import json
import math
import pickle
from pathlib import Path

POLICIES=('control','static_hf','repay_only','deposit_only','automation_style')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def same_value(left,right):
    if isinstance(left,(int,float)) and isinstance(right,(int,float)) and not isinstance(left,bool) and not isinstance(right,bool):
        return math.isclose(left,right,rel_tol=1e-8,abs_tol=1e-8)
    if isinstance(left,dict) and isinstance(right,dict):
        return left.keys()==right.keys() and all(same_value(left[k],right[k]) for k in left)
    if isinstance(left,list) and isinstance(right,list):
        return len(left)==len(right) and all(same_value(a,b) for a,b in zip(left,right))
    return left==right

def control_state(arm):
    checkpoint=arm['checkpoint_state']
    return {key:checkpoint.get(key) for key in ('timestamp','wallet_balances','debt_balances',
        'total_collateral_usd','total_debt_usd','health_factor','effective_liquidation_threshold') if key in checkpoint}

def arm_summary(arms):
    without,with_action=(arms[k] for k in ('without','with'))
    def event_time(arm):
        elapsed=arm['liquidation_stats'].get('time_to_liquidation')
        return None if elapsed is None else float(arm['checkpoint_state']['timestamp'])+float(elapsed)
    return dict(control=bool(without['liquidation_stats']['liquidated']),
        intervention=bool(with_action['liquidation_stats']['liquidated']),
        control_event_time=event_time(without),intervention_event_time=event_time(with_action),
        control_checkpoint=control_state(without),
        checkpoint_execution_failures=len(without.get('execution_failures',[])),
        intervention_execution_failures=len(with_action.get('execution_failures',[])))

def summarize(rows):
    common=[r for r in rows if r['status']=='common']
    counts={}
    for method in ('agent',*POLICIES):
        data=collections.Counter()
        for row in common:
            result=row['methods'][method];control=result['control'];intervention=result['intervention']
            data['cases']+=1;data['control_flags']+=control;data['intervention_flags']+=intervention
            data['rescued']+=control and not intervention;data['worsened']+=not control and intervention
            if method!='agent':
                proposal=row['proposals'][method]
                data['attempted']+=bool(proposal.get('action'))
                data['proposed_capital_usd']+=float(proposal.get('capital_usd',0) or 0)
                data['intervention_execution_failures']+=result['intervention_execution_failures']
            if control and intervention and result['control_event_time'] is not None and result['intervention_event_time'] is not None:
                data['both_flagged']+=1
                data['delayed_at_least_7d']+=(result['intervention_event_time']-result['control_event_time'])>=604800
        counts[method]=dict(data)
        positives=data['control_flags']
        counts[method]['prevention_fraction']=data['rescued']/positives if positives else None
    return dict(selected=len(rows),common=len(common),exclusion_reasons=dict(collections.Counter(r['reason'] for r in rows if r['status']!='common')),
        methods=counts,common_case_sha256=hashlib.sha256(json.dumps(sorted(r['case_id']+':'+r['variant'] for r in common)).encode()).hexdigest())

def combine(study,output):
    agent_dir=study/'agent-evaluation';policy_dir=study/'policy-baselines'
    parent_manifest=json.loads((agent_dir/'manifest.json').read_text())
    policy_manifest=json.loads((policy_dir/'manifest.json').read_text())
    for key in ('code_identity','recommendations_sha256'):
        if parent_manifest[key]!=policy_manifest[key]:raise ValueError('Source identity differs: '+key)
    if parent_manifest['selected']!=1200 or policy_manifest['selected']!=1200:raise ValueError('Unexpected cohort size')
    if not (agent_dir/'complete.json').exists() or not (policy_dir/'pass-complete.json').exists():raise ValueError('Pass incomplete')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(agent_manifest_sha256=sha(agent_dir/'manifest.json'),policy_manifest_sha256=sha(policy_dir/'manifest.json'),
        agent_summary_sha256=sha(agent_dir/'summary.json'),policy_pass_sha256=sha(policy_dir/'pass-complete.json'),
        code_sha256=sha(Path(__file__)),scope='fixed 30-day stress study; saved legacy funding')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    stats={}
    with (output/'cases.jsonl').open('w') as target:
        for variant in ('core','indexed'):
            rows=[]
            for index in range(1200):
                key=f'{index:05d}-{variant}'
                agent_path=agent_dir/'cases'/(key+'.json');folder=policy_dir/'cases'/key
                row=dict(index=index,variant=variant,case_id=None,status='excluded',reason=None)
                if not agent_path.exists():row['reason']='agent_file_missing'
                else:
                    agent=json.loads(agent_path.read_text())
                    if agent.get('index')!=index or agent.get('variant')!=variant:raise ValueError('Agent case identity mismatch: '+key)
                    if not agent.get('result',{}).get('success') or set(agent.get('arms',{}))!={'without','with'}:
                        row['reason']='agent_not_successful_or_paired'
                    elif (folder/'unavailable.json').exists():row['reason']='baseline_unavailable'
                    elif (folder/'error.json').exists():row['reason']='baseline_error'
                    elif not (folder/'complete.json').exists():row['reason']='baseline_incomplete'
                    elif not (folder/'recommendations.pkl').exists():row['reason']='baseline_generation_missing'
                    else:
                        with (folder/'recommendations.pkl').open('rb') as source:pack=pickle.load(source)
                        if pack['index']!=index or pack['variant']!=variant or pack['parent_case_sha256']!=sha(agent_path):
                            raise ValueError('Baseline generation source mismatch: '+key)
                        row['case_id']=str(pack['case_id'])
                        methods={'agent':arm_summary(agent['arms'])};reason=None
                        for policy in POLICIES:
                            path=folder/(policy+'.pkl')
                            if not path.exists():reason='policy_file_missing:'+policy;break
                            with path.open('rb') as source:item=pickle.load(source)
                            if item.get('index')!=index or item.get('variant')!=variant or item.get('policy')!=policy or str(item.get('case_id'))!=row['case_id']:
                                raise ValueError('Policy case identity mismatch: '+key+':'+policy)
                            if not item.get('result',{}).get('success') or set(item.get('arms',{}))!={'without','with'}:
                                reason='policy_not_successful_or_paired:'+policy;break
                            outcome=arm_summary(item['arms'])
                            if outcome['control']!=methods['agent']['control']:
                                reason='control_flag_mismatch:'+policy;break
                            if not same_value(outcome['control_checkpoint'],methods['agent']['control_checkpoint']):
                                reason='control_checkpoint_mismatch:'+policy;break
                            methods[policy]=outcome
                        if reason:row['reason']=reason
                        else:row.update(status='common',reason=None,methods=methods,proposals=pack['proposals'])
                rows.append(row);target.write(json.dumps(row,default=str)+'\n')
                if index%100==0:print(variant,index,flush=True)
            stats[variant]=summarize(rows)
    report=dict(variants=stats,manifest=manifest,interpretation='Same recommendation ID, simulator identity, agent source, successful paired arms, and control state; fixed 30-day follow-up. Exclusions retained. Saved legacy wallet inference and variable capital/action triggers remain limitations. Outcome is simulated liquidation eligibility, not observed execution.')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({v:dict(common=s['common'],exclusions=s['exclusion_reasons'],methods={k:dict(rescued=x['rescued'],control_flags=x['control_flags']) for k,x in s['methods'].items()}) for v,s in stats.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();combine(a.study.resolve(),a.output.resolve())
