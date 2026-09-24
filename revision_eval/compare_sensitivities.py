import argparse,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
ref=json.loads((a.root/'main/case-metrics.json').read_text());key=lambda r:(r['case_id'],r['cohort'],r['variant'],r['policy']);reference={key(r):r for r in ref};output={}
for run in sorted((a.root/'sensitivities').glob('*')):
 if not (run/'case-metrics.json').exists():continue
 rows=json.loads((run/'case-metrics.json').read_text());config=json.loads((run/'manifest.json').read_text())['config'];groups={}
 for row in rows:
  old=reference.get(key(row))
  if old is None:continue
  g=groups.setdefault(row['cohort']+':'+row['policy'],collections.Counter());g['matched']+=1
  for side,r in [('reference',old),('scenario',row)]:
   g[side+'_baseline']+=r['baseline'];g[side+'_rescued']+=r['baseline'] and not r['intervention'];g[side+'_worsened']+=r['intervention'] and not r['baseline']
 output[run.name]=dict(config=config,groups=groups,complete=(run/'complete.json').exists(),comparison_scope='common completed cases; see each summary for all-eligible failures; warnings are not protocol outcomes')
dust={}
for threshold in [0,1,10]:
 for r in ref:
  if r['debt_usd']<threshold:continue
  k=str(threshold)+':'+r['cohort']+':'+r['variant']+':'+r['policy'];c=dust.setdefault(k,collections.Counter());c['paired']+=1;c['baseline']+=r['baseline'];c['rescued']+=r['baseline'] and not r['intervention'];c['worsened']+=r['intervention'] and not r['baseline']
(a.root/'sensitivity-comparison.json').write_text(json.dumps(dict(scenarios=output,checkpoint_debt_filter_usd=dust),indent=2))
