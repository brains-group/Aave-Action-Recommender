"""Read-only progress for the frozen tuned-bot stress evaluation."""
import argparse,json,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def show(run):
 m=json.loads((run/'manifest.json').read_text());jobs=json.loads((run/'jobs.json').read_text())
 p=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
 completed={f.stem for f in (run/'cases').glob('*.pkl')};by_variant=Counter(j['variant'] for j in jobs if j['case_id'] in completed)
 remaining=[j for j in jobs if j['case_id'] not in completed];actions=sum(j['proposal']['action'] is not None for j in remaining)
 active=[]
 for file in (run/'active').glob('*.json'):
  try:
   item=json.loads(file.read_text())
   if time.time()-item.get('heartbeat',0)<20 and item.get('phase')!='idle':active.append(item)
  except (OSError,ValueError):pass
 timings=[json.loads(f.read_text())['seconds'] for f in (run/'simulation-timings').glob('*.json')]
 failures=list((run/'failures').glob('*.json'));complete=(run/'complete.json').exists()
 print(f"Tuned rule-based bot | frozen HF {m['configuration']} | {'COMPLETE' if complete else 'RUNNING'}")
 print(f"Cases: {len(completed):,}/{len(jobs):,} completed; {len(remaining):,} remaining; {len(failures)} current errors")
 print('Core: %s/%s | Indexed: %s/%s | Active workers: %s/%s' % (by_variant['core'],m['cases']['core'],by_variant['indexed'],m['cases']['indexed'],len(active),p.get('workers',m['workers'])))
 print(f"Remaining action proposals: {actions}; new simulations completed: {len(timings)}")
 print('Elapsed: %.1f minutes' % (p.get('elapsed_seconds',0)/60))
 if len(timings)>=10 and active:
  estimate=sum(timings)/len(timings)*actions/p.get('workers',m['workers'])
  print('Approximate remaining replay work: %.1f minutes; cache reuse may shorten it and slow cases may extend it.' % (estimate/60))
 elif not complete:print('Runtime estimate: waiting for enough new intervention timings; early abstention throughput is not a reliable ETA.')
 if p.get('heartbeat') and time.time()-p['heartbeat']>30 and not complete:print('Coordinator heartbeat is stale; inspect runner.log.')
 if complete:
  s=json.loads((run/'matched-results.json').read_text())
  for variant,policies in s['by_variant'].items():
   for policy in ('agent','tuned_rule_bot'):
    v=policies[policy];print(f"{variant} {policy}: {v['prevented']}/{v['control_liquidations']} prevented; {v['worsened']} worsened")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'outputs/tuned-rule-stress-20261013');p.add_argument('--watch',type=float);a=p.parse_args()
 while True:
  show(a.run.resolve())
  if not a.watch or (a.run/'complete.json').exists():break
  print(flush=True);time.sleep(max(1,a.watch))
if __name__=='__main__':main()
