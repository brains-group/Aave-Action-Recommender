"""Summarize all selected cases, keeping errors/abstentions and paired denominators."""
import argparse,pickle,json,collections,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();groups={};details=[]
 manifest=json.loads((a.run/'manifest.json').read_text());cohort_ids={};traces=[]
 for folder in sorted((a.run/'cases').iterdir()):
  if not folder.is_dir():continue
  variant=folder.name.split('-')[-1];generation=folder/'recommendations.pkl'
  if not generation.exists():continue
  pack=pickle.load(generation.open('rb'));cohort=pack['cohort'];state=pack['state'];cohort_ids.setdefault(cohort,set()).add(pack['case_id'])
  for policy in pack['recommendations']:
   group=groups.setdefault(cohort+':'+variant+':'+policy,collections.Counter());group['generated']+=1
   rec,info=pack['recommendations'][policy];group['abstained']+=bool(rec.get('abstained'));group['generation_error']+=bool(rec.get('generation_error'))
   path=folder/(policy+'.pkl')
   if not path.exists():continue
   row=pickle.load(path.open('rb'));group['evaluated']+=1;group['script_success']+=bool(row['result'].get('success'))
   arms=row['arms'];group['execution_error_cases']+=any(arm.get('execution_failures') for arm in arms.values())
   if set(arms)!={'without','with'}:group['missing_paired_arm']+=1;continue
   b=arms['without']['liquidation_stats'];i=arms['with']['liquidation_stats'];bl=bool(b['liquidated']);il=bool(i['liquidated'])
   group['paired']+=1;group['baseline_liquidated']+=bl;group['intervention_liquidated']+=il;group['rescued']+=bl and not il;group['worsened']+=il and not bl
   attempted=not rec.get('abstained') and not rec.get('generation_error');failed=any(float(tx.get('timestamp',-1))==float(rec['timestamp']) for tx in arms['with'].get('execution_failures',[]));feasible=attempted and not failed
   capital=float(rec.get('amountUSD',0)) if attempted else 0.;group['attempted']+=attempted;group['feasible_action']+=feasible;group['proposed_capital_usd']+=capital;group['executed_capital_usd']+=float(row.get('applied_recommendation',{}).get('amountUSD',capital)) if feasible else 0
   # Compare absolute times because the arms start at different action timestamps.
   delay=None
   if bl and il and b.get('time_to_liquidation') is not None and i.get('time_to_liquidation') is not None:
    delay=(arms['with']['checkpoint_state']['timestamp']+i['time_to_liquidation'])-(arms['without']['checkpoint_state']['timestamp']+b['time_to_liquidation']);group['delayed_1d']+=delay>=86400;group['delayed_7d']+=delay>=604800
   observed=pack.get('observed_liquidation_timestamp');pre_action=observed is not None and observed<float(rec['timestamp'])
   observed_in_window=observed is not None and float(rec['timestamp'])<=observed<=row['followup_end']
   if policy=='control' and not pre_action and 'observed_liquidation_timestamp' in pack:
    group['execution_label_'+('tp' if bl and observed_in_window else 'fp' if bl else 'fn' if observed_in_window else 'tn')]+=1
    if bl!=observed_in_window and len(traces)<20:
     traces.append(dict(case_id=pack['case_id'],cohort=cohort,variant=variant,eligible_flag=bl,observed_execution=observed_in_window,checkpoint=pack['checkpoint'],note='Eligibility versus observed execution mismatch; not independent proof of an incorrect eligibility rule'))
   group['observed_liquidation_before_action']+=pre_action
   details.append(dict(case_id=pack['case_id'],cohort=cohort,variant=variant,policy=policy,baseline=bl,intervention=il,feasible=bool(feasible),capital_usd=capital,delay_seconds=delay,debt_usd=state['total_debt_usd'],activity_30d=state.get('activity_30d'),checkpoint=state['timestamp'],dust_below_1=state['total_debt_usd']<1,dust_below_10=state['total_debt_usd']<10))
 for key,counts in groups.items():
  counts['eligible_cases']=sum(c==key.split(':')[0] for c in manifest['cohort_membership'].values())
  counts['unavailable_cases']=counts['eligible_cases']-counts['paired']
  counts['illustrative_gas_cost_usd']={str(fee):fee*counts['attempted'] for fee in [0,.1,1,5]}
  counts['prevention_fraction']=counts['rescued']/counts['baseline_liquidated'] if counts['baseline_liquidated'] else None
  n=counts['baseline_liquidated']
  if n:
   phat=counts['rescued']/n;z=1.95996398454;center=(phat+z*z/(2*n))/(1+z*z/n);width=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
   counts['prevention_wilson_95']=[center-width,center+width]
  else:counts['prevention_wilson_95']=None
  counts['feasible_intervention_rate']=counts['feasible_action']/counts['attempted'] if counts['attempted'] else None
 result={'groups':groups,'complete':(a.run/'complete.json').exists(),'generation_complete':(a.run/'generation-complete.json').exists(),'selected_cases':len(manifest['case_ids']),'case_generation_failures':len(list((a.run/'cases').glob('*/error.json'))),'gas_slippage':'unavailable; capital is not a transaction cost','followup':'seven-day common end; projected positions/prices, no replay of later recorded user actions','funding':'same retrospective profile wallet assumption for all policies; sensitivity multiplier in manifest'}
 (a.run/'mismatch-traces.json').write_text(json.dumps(traces,indent=2,default=str));(a.run/'summary.json').write_text(json.dumps(result,indent=2));(a.run/'case-metrics.json').write_text(json.dumps(details,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
