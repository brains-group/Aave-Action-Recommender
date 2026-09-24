"""Summarize completed paired cases; always label incomplete runs explicitly."""
import argparse,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args();records={};summary={}
for path in (a.run_dir/'cases').glob('*.json'):
 r=json.loads(path.read_text());records[(r['index'],r['variant'])]=r
for variant in ['core','indexed']:
 rows=[r for (i,v),r in records.items() if v==variant];s={'completed':len(rows),'successful':0,'paired_arms':0,'skips_errors':collections.Counter(),'all_paired':collections.Counter(),'script_filtered':collections.Counter(),'hf_only':collections.Counter()}
 for r in rows:
  result=r['result'];arms=r['arms']
  if not result.get('success'):s['skips_errors'][result.get('error') or 'recommendation not feasible / skipped']+=1
  else:
   s['successful']+=1
   for name,key in [('script_filtered','stats_updates'),('hf_only','stats_updates_hf_only')]:
    for k in ['processed','liquidated_without','liquidated_with','improved','worsened','delayed_liquidations_7d']:s[name][k]+=result.get(key,{}).get('overall',{}).get(k,0)
  if 'without' in arms and 'with' in arms:
   s['paired_arms']+=1;b=arms['without']['liquidation_stats'];i=arms['with']['liquidation_stats'];c=s['all_paired'];c['liquidated_without']+=bool(b['liquidated']);c['liquidated_with']+=bool(i['liquidated']);c['improved']+=bool(b['liquidated'] and not i['liquidated']);c['worsened']+=bool(i['liquidated'] and not b['liquidated'])
 for name in ['all_paired','script_filtered','hf_only']:
  c=s[name];den=c.get('liquidated_without',0);c['prevention_fraction']=c.get('improved',0)/den if den else None
 summary[variant]=s
common={i for i,v in records if all((i,w) in records and set(records[(i,w)]['arms'])=={'without','with'} for w in ['core','indexed'])}
summary['matched_completed_recommendations']=len(common)
summary['matched_all_paired']={}
for variant in ['core','indexed']:
 c=collections.Counter()
 for i in common:
  arms=records[(i,variant)]['arms'];b=arms['without']['liquidation_stats']['liquidated'];z=arms['with']['liquidation_stats']['liquidated'];c['baseline']+=b;c['intervention']+=z;c['rescued']+=b and not z;c['worsened']+=z and not b
 c['prevention_fraction']=c['rescued']/c['baseline'] if c['baseline'] else None;summary['matched_all_paired'][variant]=c
summary['complete']=(a.run_dir/'complete.json').exists();summary['manifest']=json.loads((a.run_dir/'manifest.json').read_text());(a.run_dir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
