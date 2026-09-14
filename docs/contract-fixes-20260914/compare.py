import json,collections
from pathlib import Path
root=Path('/home/spadef/data/craft-soc/data/evaluation');out={}
for variant in ['core','indexed']:
 totals=[collections.Counter(),collections.Counter()];indices=[]
 for i in range(12000):
  paths=[root/'recommendations-20260910'/'cases'/f'{i:05d}-{variant}.json',root/'refreshed-recommendations-20260911/evaluation/cases'/f'{i:05d}-{variant}.json']
  if not all(p.exists() for p in paths):continue
  rows=[json.loads(p.read_text()) for p in paths]
  values=[r['result'].get('stats_updates_hf_only',{}).get('overall',{}) if r['result'].get('success') else {} for r in rows]
  if not all(v.get('processed') for v in values):continue
  assert rows[0]['arms']['without']['user_address']==rows[1]['arms']['without']['user_address']
  indices.append(i)
  for total,v in zip(totals,values):
   total.update({k:v.get(k,0) for k in ['processed','liquidated_without','liquidated_with','improved','worsened','delayed_liquidations_7d']})
 out[variant]={'indices':indices,'old':totals[0],'refreshed':totals[1],'limitation':'Intersection of two outcome-filtered HF-only subsets; not an unbiased cohort.'}
Path(__file__).with_name('matched-review.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:{j:v for j,v in d.items() if j!='indices'} for k,d in out.items()},indent=2))
