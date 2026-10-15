"""Offline matched proposal/wallet-change audit; no simulations or policy tuning."""
import argparse,collections,json,math,pickle,statistics,time
from pathlib import Path
from . import audit_baseline_tuning as audit
from . import run_bayesian_tuning as driver
from .combine_completed_stress import control_state,same_value
from utils.recommendation_funding import funding_choice,NoFeasibleAction

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/high-risk-shortest-20260925/studies/fixed-2592000-seconds-trained-v1'
DEFAULT=ROOT/'outputs/bayesian-agent-gap-audit-20261013'


def asdict(rec):return rec.to_dict() if hasattr(rec,'to_dict') else dict(rec)

def selected_rec(rec):
    return {k:rec.get(k) for k in ['user','timestamp','Index Event','type','reserve','symbol','amount','amountUSD','priceInUSD','abstained','abstention_reason']}

def price(state,symbol):return float(state['assets'][symbol]['price_usd'])

def wallet_change(arms):
    before=arms['without']['checkpoint_state'];after=arms['with']['checkpoint_state'];a=before['wallet_balances'];b=after['wallet_balances']
    changes={k:float(a.get(k,0))-float(b.get(k,0)) for k in set(a)|set(b)}
    # Preserve the raw differences. A relative tolerance marks practical changes
    # at very large saved wallet balances; it does not change recorded outcomes.
    changed={k:v for k,v in changes.items() if abs(v)>max(1e-8,1e-10*max(abs(float(a.get(k,0))),abs(float(b.get(k,0)))))}
    missing=[k for k,v in changed.items() if v>0 and (k not in before.get('assets',{}) or not math.isfinite(price(before,k)) or price(before,k)<=0)]
    amount=None if missing else sum(v*price(before,k) for k,v in changed.items() if v>0)
    return dict(wallet_debits_units={k:v for k,v in changed.items() if v>0},wallet_credits_units={k:-v for k,v in changed.items() if v<0},
        checkpoint_price_wallet_debit_usd=amount,missing_prices=missing,
        before_timestamp=before['timestamp'],after_timestamp=after['timestamp'])


def stats(values):
    return dict(cases=len(values),total=sum(values),median=statistics.median(values),p90=sorted(values)[int(.9*(len(values)-1))])


def run(output):
    output.mkdir(parents=True,exist_ok=False);run=driver.DEFAULT
    manifest=json.loads((run/'manifest.json').read_text());driver.check_inputs(run,manifest)
    assert (run/'verification.json').exists()
    main_manifest=json.loads((driver.OLD/'manifest.json').read_text())
    mainpath=next(Path(p) for p in main_manifest['original_inputs'] if p.endswith('/recommendations.pkl'))
    stresspath=STUDY/'generation-core/recommendations.pkl'
    sm=json.loads((STUDY/'agent-evaluation/manifest.json').read_text())
    assert audit.sha(stresspath)==sm['recommendations_sha256']
    assert audit.sha(mainpath)==main_manifest['original_inputs'][str(mainpath)]
    main=pickle.load(mainpath.open('rb'));stress=pickle.load(stresspath.open('rb'));generation=json.loads((STUDY/'generation-core/manifest.json').read_text())
    inputs={str(p):audit.sha(p) for p in [mainpath,stresspath,STUDY/'agent-evaluation/manifest.json',STUDY/'generation-core/manifest.json',run/'verification.json',run/'frozen-setting.json',run/'core-main-matched-cases.jsonl',run/'indexed-main-matched-cases.jsonl',run/'stress-matched-cases.jsonl']}
    groups=collections.defaultdict(list);misses=[];receipts={};started=time.time()
    for cohort,files in [('main',['core-main-matched-cases.jsonl','indexed-main-matched-cases.jsonl']),('stress',['stress-matched-cases.jsonl'])]:
        for file in files:
            for row in map(json.loads,(run/file).open()):
                variant=row.get('variant','core' if file.startswith('core') else 'indexed')
                if cohort=='main':
                    index=int(row['case_id'].split('-')[0]);pair=main[str(index)]
                    assert str(pair[0]['user']).lower()==row['user'].lower()
                else:
                    index=row['index'];pair=stress[row['case_id']]
                    assert list(stress)[index]==row['case_id']
                rec=asdict(pair[0]);info=pair[1];amount=float(rec.get('amountUSD',0) or 0)
                assert math.isfinite(amount) and amount>=0
                item=dict(case_id=row['case_id'],index=index,cohort=cohort,variant=variant,
                    user=rec['user'],control=row['methods']['agent']['control'],agent_prevented=row['methods']['agent']['control'] and not row['methods']['agent']['intervention'],
                    agent_proposed_capital_usd=amount,agent_abstained=bool(rec.get('abstained',False)),
                    bot_proposed_capital_usd=row['methods']['tuned_rule_bot']['proposed_capital_usd'],
                    bot_prevented=row['methods']['tuned_rule_bot']['control'] and not row['methods']['tuned_rule_bot']['intervention'])
                if cohort=='stress':
                    key=f'{index:05d}-{variant}';ap=STUDY/'agent-evaluation/cases'/(key+'.json');bp=run/'stress-engine/cases'/(key+'.pkl')
                    agent=json.loads(ap.read_text());bot=pickle.load(bp.open('rb'))
                    assert same_value(control_state(agent['arms']['without']),control_state(bot['arms']['without']))
                    assert agent['result']['success'] and bot['result']['success']
                    item['agent_wallet_change']=wallet_change(agent['arms']);item['bot_wallet_change']=wallet_change(bot['arms'])
                    action_time=float(rec['timestamp'])
                    item['agent_action_failures']=[f for f in agent['arms']['with'].get('execution_failures',[]) if abs(float(f.get('timestamp',-1))-action_time)<1e-6]
                    if row['methods']['agent']['control'] and row['methods']['agent']['intervention']:
                        receipts[str(ap)]=audit.sha(ap);receipts[str(bp)]=audit.sha(bp)
                        before=agent['arms']['without']['checkpoint_state'];after=agent['arms']['with']['checkpoint_state'];t=float(rec['timestamp'])
                        failed=[f for f in agent['arms']['with'].get('execution_failures',[]) if abs(float(f.get('timestamp',-1))-t)<1e-6]
                        if item['agent_abstained']:reason='core_generated_abstention'
                        elif failed:reason='recommended_action_execution_failure'
                        elif not item['agent_wallet_change']['wallet_debits_units']:reason='no_wallet_debit_observed'
                        else:reason='action_applied_but_liquidation_remains'
                        trace=dict(item,classification=reason,agent_recommendation=selected_rec(rec),agent_risk_flag=info.get('is_at_risk'),agent_immediate_risk=info.get('is_at_immediate_risk'),
                            agent_action_failures=failed,agent_pre_state=before,agent_post_state=after,
                            bot_transaction=bot['transaction'],bot_proposal=bot['proposal'],bot_post_state=bot['arms']['with']['checkpoint_state'],
                            agent_liquidation={k:agent['arms']['with']['liquidation_stats'].get(k) for k in ['liquidated','time_to_liquidation','liquidation_reason']},
                            control_liquidation={k:agent['arms']['without']['liquidation_stats'].get(k) for k in ['liquidated','time_to_liquidation','liquidation_reason']},
                            source_agent=str(ap),source_bot=str(bp),generation_mode=generation['checkpoint_mode'])
                        cp=STUDY/'agent-evaluation/cases'/f'{index:05d}-core.json'
                        core=json.loads(cp.read_text())['arms']['without']['checkpoint_state'];receipts[str(cp)]=audit.sha(cp)
                        if not item['agent_abstained']:
                            symbol=rec.get('symbol') or rec['reserve'];maximum=float(core['wallet_balances'].get(symbol,0))
                            if str(rec['Index Event']).lower()=='repay':maximum=min(maximum,float(core['debt_balances'].get(symbol,0)))
                            minimum=min(max(50/float(rec['priceInUSD']),maximum/12),maximum)
                            trace['core_generation_amount_recheck']=dict(minimum=minimum,maximum=maximum,selected_amount=float(rec['amount']),
                                selected_initial_amount=math.isclose(float(rec['amount']),minimum,rel_tol=1e-8,abs_tol=1e-8),
                                selected_at_maximum=math.isclose(float(rec['amount']),maximum,rel_tol=1e-8,abs_tol=1e-8),
                                risk_loop_can_increase=float(rec['amount'])<maximum)
                        if item['agent_abstained']:
                            trace['generation_core_state']=core
                            for name,state in [('core',core),('indexed',before)]:
                                try:
                                    choice=funding_choice(state,'Repay',t-600,lambda k:price(state,k))
                                    trace[name+'_funding_recheck']=dict(asset=choice[0],minimum=choice[1],maximum=choice[2])
                                except NoFeasibleAction as e:trace[name+'_funding_recheck']=dict(error=str(e))
                        misses.append(trace)
                groups[(cohort,variant)].append(item)
            print(cohort,variant,'processed',len(groups[(cohort,variant)]),flush=True)
    summary={}
    for (cohort,variant),items in groups.items():
        result=dict(cases=len(items),positive_controls=sum(x['control'] for x in items),
            agent_proposed=stats([x['agent_proposed_capital_usd'] for x in items]),bot_proposed=stats([x['bot_proposed_capital_usd'] for x in items]),
            agent_positive_proposals=sum(x['agent_proposed_capital_usd']>0 for x in items),bot_positive_proposals=sum(x['bot_proposed_capital_usd']>0 for x in items),
            agent_proposed_on_positive_controls=sum(x['agent_proposed_capital_usd'] for x in items if x['control']),bot_proposed_on_positive_controls=sum(x['bot_proposed_capital_usd'] for x in items if x['control']))
        if cohort=='stress':
            for policy in ('agent','bot'):
                changes=[x[policy+'_wallet_change'] for x in items]
                good=[x['checkpoint_price_wallet_debit_usd'] for x in changes if x['checkpoint_price_wallet_debit_usd'] is not None]
                result[policy+'_wallet_debit']=dict(stats(good),unknown_prices=len(changes)-len(good),multiple_debit_assets=sum(len(x['wallet_debits_units'])>1 for x in changes),credited_wallet_rows=sum(bool(x['wallet_credits_units']) for x in changes),nonzero_rows=sum(x>0 for x in good))
                result[policy+'_wallet_debit_on_positive_controls_usd']=sum(x[policy+'_wallet_change']['checkpoint_price_wallet_debit_usd'] or 0 for x in items if x['control'])
            result['agent_action_failure_rows']=sum(bool(x['agent_action_failures']) for x in items)
            result['agent_positive_control_action_failure_rows']=sum(bool(x['agent_action_failures']) and x['control'] for x in items)
        summary[cohort+'_'+variant]=result
    (output/'cases.jsonl').write_text(''.join(json.dumps(x,default=str)+'\n' for group in groups.values() for x in group))
    (output/'missed-stress-traces.json').write_text(json.dumps(misses,indent=2,default=str))
    (output/'summary.json').write_text(json.dumps(dict(variants=summary,missed_case_classification=dict(collections.Counter(x['classification'] for x in misses)),
        missed_actions_risk_flags=dict(collections.Counter(str(x['agent_risk_flag']) for x in misses)),core_generated_recommendations=generation['checkpoint_mode']=='core',
        inputs=inputs,missed_case_input_hashes=receipts,script_sha256=audit.sha(__file__),simulation_calls=0,seconds=time.time()-started),indent=2))
    print(json.dumps(summary,indent=2))
    print('Miss classifications',dict(collections.Counter(x['classification'] for x in misses)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);args=p.parse_args();run(args.output.resolve())
