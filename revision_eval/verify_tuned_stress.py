"""Independently verify completed frozen-bot stress results; never run simulation."""
import argparse,json,pickle
from collections import Counter
from pathlib import Path
from . import audit_baseline_tuning as audit
from . import run_baseline_tuning as common
from .combine_completed_stress import control_state,same_value
from .run_tuned_stress import CALIBRATION,ROOT,RUN

def verify(run):
 m=json.loads((run/'manifest.json').read_text());jobs=json.loads((run/'jobs.json').read_text())
 assert (run/'complete.json').exists(),'Run is not complete'
 for path,digest in m['inputs'].items():assert audit.sha(path)==digest,path
 assert audit.sha(run/'jobs.json')==m['jobs_sha256']
 assert audit.sha(CALIBRATION/'calibration-summary.json')==m['frozen_setting']['calibration_summary_sha256']
 rows=[json.loads(line) for line in (run/'matched-cases.jsonl').open()]
 assert len(rows)==len(jobs)==1695
 assert {(r['index'],r['variant'],r['case_id']) for r in rows}=={(j['previous']['index'],j['variant'],j['economic_case_id']) for j in jobs}
 assert Counter(r['variant'] for r in rows)==m['cases']
 for job in jobs:
  value=pickle.load((run/'cases'/(job['case_id']+'.pkl')).open('rb'))
  assert value['result']['success'] and set(value['arms'])=={'without','with'}
  assert same_value(control_state(value['arms']['without']),job['previous']['methods']['agent']['control_checkpoint'])
  assert common.outcome(value['arms']['without'])==job['previous']['methods']['agent']['control']
  assert value['recommendation'][0]['type']==value['recommendation'][0]['Index Event']
  if value['transaction'] is not None:
   assert value['transaction']['action'].lower()==job['proposal']['action'].lower()
  assert value['configuration']==m['configuration']
 summary=json.loads((run/'matched-results.json').read_text())
 for variant in ('core','indexed'):
  selected=[r for r in rows if r['variant']==variant]
  for policy in ('agent',*audit.POLICIES,'tuned_rule_bot'):
   assert common.count_outcomes([r['methods'][policy] for r in selected])==summary['by_variant'][variant][policy]
 assert not list((run/'failures').glob('*.json'))
 report=dict(all_cases_verified=True,case_count=len(rows),calibration_unchanged=True,controls_match=True,
             types_match=True,summaries_recomputed=True,exact_original_case_ids=True,by_variant=summary['by_variant'])
 common.save(run/'verification.json',report)
 doc=ROOT/'docs/revision-eval/TUNED_RULE_STRESS_RESULTS_20261013.md'
 text=['# Tuned rule-based bot: verified stress-test results\n\nThe fixed 1.25/1.60 configuration was tested without rerunning calibration on the exact existing matched stress cases. Original controls, funding, prices, simulation revision and 30-day evaluator were retained.\n\n| Reconstruction | Policy | Cases | Control liquidations | Prevented | Prevention rate | Worsened |\n|---|---|---:|---:|---:|---:|---:|\n']
 for variant in ('core','indexed'):
  for policy in ('agent',*audit.POLICIES,'tuned_rule_bot'):
   v=summary['by_variant'][variant][policy]
   text.append(f"| {variant} | {policy} | {v['cases']} | {v['control_liquidations']} | {v['prevented']} | {100*v['prevention_rate']:.2f}% | {v['worsened']} |\n")
 text.append('\nIndependent verification recomputed every method summary from case records, confirmed exact old matched identities and all pre-event control states, checked all successful paired arms and recommendation/action types, and rechecked the unchanged calibration hashes. Authoritative files: `outputs/tuned-rule-stress-20261013/{matched-results.json,matched-cases.jsonl,verification.json,complete.json}`. Manuscript stress numbers must be updated from these completed results with changes-package tracking; no such results were assumed before completion.\n')
 doc.write_text(''.join(text));print(json.dumps({v:{p:s for p,s in values.items() if p in ('agent','tuned_rule_bot')} for v,values in summary['by_variant'].items()},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);a=p.parse_args();verify(a.run.resolve())
